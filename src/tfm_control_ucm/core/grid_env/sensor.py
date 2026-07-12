from gc import get_debug

import numpy as np
from typing import Iterable, Tuple, Union

import torch
from turtledemo.chaos import distance
from tfm_control_ucm.core.grid_env.map import GridMap
from ...utils.utils import get_device
import numpy as np
from typing import Iterable, Tuple, Union

class GridLidar:
    """
    A 2D LIDAR sensor that casts multiple rays in a configurable field of view.
    """
    def __init__(
        self,
        *,
        num_rays: int = 10,
        max_range: int = 10,
        fov: float = np.pi,
        noise_std: float = 0.001,
        with_cache: bool = True,
        device: Union[torch.device,str] = "cpu"
    ):
        if not isinstance(num_rays, np.number|int): raise TypeError("The num_rays must be an integer")
        if not isinstance(max_range, np.number|int): raise TypeError("The max_range must be an integer")
        if not isinstance(fov, np.number|float): raise TypeError("The fov must be a float")
        if not isinstance(noise_std, np.number|float): raise TypeError("The noise_std must be a float")
        if not isinstance(with_cache, bool): raise TypeError("The with_cache must be a boolean")
        if not isinstance(device,(str,torch.device)): raise TypeError("The device must be a torch.device")


        if num_rays <= 0: raise ValueError("The num_rays must be greater than zero")
        if max_range <= 0: raise ValueError("The max_range must be greater than zero")
        if fov <= 0: raise ValueError("The fov must be greater than zero")
        if noise_std <= 0: raise ValueError("The noise_std must be greater than zero")

        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_rad = fov
        self.noise_std = noise_std
        self.with_cache = with_cache
        if isinstance(device,str): self.device = torch.device(device)
        else: self.device = device
        
        angles = torch.linspace(-fov/2,fov/2, num_rays, device=device)
        self.angles_rad = angles.unsqueeze(1)

        if self.with_cache:
            self.dist_grid = torch.arange(1, max_range+1, device=device, dtype=torch.float32)
            self.dr_base = torch.round(-torch.sin(self.angles_rad) * self.dist_grid).type(torch.int32) # [R, D]
            self.dc_base = torch.round(torch.cos(self.angles_rad) * self.dist_grid).to(torch.int32)  # [R, D]

        # Diccionario de Lookup Tables para los mapas procesados
        self._map_caches = {}
    

    def get_config(self):
        return {'num_rays':self.num_rays, 'max_range':self.max_range, 'fov': self.fov_rad, 'noise_std': self.noise_std, 'with_cache': self.with_cache, 'device': str(self.device)}
    
    @classmethod
    def config_keys(cls):
        return ['num_rays', 'max_range', 'fov', 'noise_std', 'with_cache', 'device']

    def _build_full_map_cache(self, gridmap) -> torch.Tensor:
        height, width = gridmap.grid.shape
        cache_table = torch.full((height, width, self.num_rays), self.max_range, dtype=torch.float32, device=self.device)
        
        # Ensure grid is on the correct device for fast indexing
        grid_device = gridmap.grid.to(self.device)
        
        free_rows, free_cols = torch.where(grid_device != gridmap.OBSTACLE)
        if len(free_rows) == 0:
            return cache_table

        dist = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
        # self.angles_rad is (N, 1), dist is (D,) -> Product is (N, D)
        dr_base = torch.round(-torch.sin(self.angles_rad) * dist).type(torch.int32)
        dc_base = torch.round(torch.cos(self.angles_rad) * dist).type(torch.int32)

        chunk_size = 5000 
        for i in range(0, len(free_rows), chunk_size):
            r_chunk = free_rows[i : i + chunk_size]
            c_chunk = free_cols[i : i + chunk_size]
            
            # (C, 1, 1) + (1, N, D) -> (C, N, D)
            cc_global = c_chunk[:, None, None] + dc_base[None, :, :]
            rr_global = r_chunk[:, None, None] + dr_base[None, :, :]
            
            valid_mask = (rr_global >= 0) & (rr_global < height) & (cc_global >= 0) & (cc_global < width)
            rr_clamped = torch.clip(rr_global, 0, height - 1)
            cc_clamped = torch.clip(cc_global, 0, width - 1)
            
            map_hits = grid_device[rr_clamped, cc_clamped] == gridmap.OBSTACLE
            
            hits = torch.ones_like(rr_global, dtype=torch.bool)
            hits[valid_mask] = map_hits[valid_mask]
            
            # argmax over the distance dimension (now dim=2)
            hit_indices = torch.argmax(hits.to(torch.int32), dim=2)
            no_hits = ~torch.any(hits, dim=2)
            
            dists = (hit_indices + 1).type(torch.float32)
            dists[no_hits] = float(self.max_range)
            
            cache_table[r_chunk, c_chunk] = dists
            
        return cache_table

    def scan(self, gridmap, position: np.ndarray|torch.Tensor|Iterable[int], orientation: np.ndarray|torch.Tensor ) -> torch.Tensor:
        # Standardize inputs to torch tensors on self.device
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, device=self.device, dtype=torch.float32)
        else:
            position = position.to(self.device).float()
            
        if not isinstance(orientation, torch.Tensor):
            orientation = torch.tensor(orientation, device=self.device, dtype=torch.float32)
        else:
            orientation = orientation.to(self.device).float()

        # Opción 1: Extracción de Caché vectorizada
        if self.with_cache:
            if gridmap.name not in self._map_caches:
                self._map_caches[gridmap.name] = self._build_full_map_cache(gridmap)
                
            c, r = position.long()
            hit_distances = self._map_caches[gridmap.name][r, c].clone()
            angles = self.angles_rad.flatten()
            
        # Opción 2: Computación Vectorizada On-The-Fly
        else:
            dist = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
            # Shift rays by orientation
            angles = (self.angles_rad + orientation).flatten()
            
            dr = torch.round(-torch.sin(angles)[:, None] * dist).type(torch.int32)
            dc = torch.round(torch.cos(angles)[:, None] * dist).type(torch.int32)

            cc_global = position[0].long() + dc
            rr_global = position[1].long() + dr

            height, width = gridmap.grid.shape
            valid_mask = (rr_global >= 0) & (rr_global < height) & (cc_global >= 0) & (cc_global < width)
            
            grid_device = gridmap.grid.to(self.device)
            hits = torch.ones_like(rr_global, dtype=torch.bool)
            hits[valid_mask] = (grid_device[rr_global[valid_mask], cc_global[valid_mask]] == gridmap.OBSTACLE)

            hit_indices = torch.argmax(hits.to(torch.int32), dim=1).to(self.device)
            no_hits = ~torch.any(hits, dim=1)
            
            hit_distances = (hit_indices + 1).type(torch.float32)
            # Ensure max_range is assigned correctly to rays with no hits
            hit_distances = torch.where(no_hits, torch.tensor(float(self.max_range), device=self.device), hit_distances)

        # Aplicación estocástica en tiempo real
        if self.noise_std > 0:
            noise = torch.normal(0, self.noise_std, size=hit_distances.shape, device=self.device)
            # The noise must be zero if there is no hit (i.e., distance == max_range)
            noise = torch.where(hit_distances == float(self.max_range), torch.zeros_like(noise), noise)
            hit_distances = torch.clip(hit_distances + noise, 0.0, float(self.max_range))

        # Mantener formato matriz [distancia, ángulo]
        return torch.column_stack((hit_distances, angles))



class GridLidar_2D(GridLidar):
    """
    A 2D LIDAR sensor that casts multiple rays in a configurable field of view.
    Returns a 2D local occupancy grid image.
    """
    def __init__(
        self,
        *,
        num_rays: int = 10,
        max_range: int = 10,
        fov: float = np.pi,
        noise_std: float = 0.001,
        device: Union[torch.device,str] = "cpu"
    ):
        super().__init__(num_rays=num_rays, max_range=max_range, fov=fov, noise_std=noise_std, with_cache=False, device=device)

    
    def _perform_scan(self, gridmap: GridMap, position: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        grid_size = 2 * self.max_range + 1
        lidar_image = torch.full((grid_size, grid_size), -1, dtype=torch.int8, device=self.device)
        center = self.max_range
        lidar_image[center, center] = 0

        # Crear matrices de distancia y ángulos
        # Calcular coordenadas relativas (dr, dc) para todos los rayos y distancias simultáneamente
        dist = torch.arange(1, self.max_range + 1, dtype=torch.float32, device=self.device)
        # angles is (N, 1), dist is (D,) -> Result is (N, D)
        dr = torch.round(-torch.sin(angles) * dist).type(torch.int32)
        dc = torch.round(torch.cos(angles) * dist).type(torch.int32)

        # Convertir a coordenadas globales del mapa (usando tensores para evitar sincronización CPU-GPU)
        cc_global = position[0].long() + dc
        rr_global = position[1].long() + dr

        # Asegurar que el mapa está en el dispositivo correcto
        grid_device = gridmap.grid.to(self.device)
        height, width = grid_device.shape

        # Máscara para descartar coordenadas fuera de los límites del mapa
        valid_mask = (rr_global >= 0) & (rr_global < height) & (cc_global >= 0) & (cc_global < width)
        
        # Evaluar colisiones de forma vectorizada
        hits = torch.ones_like(rr_global, dtype=torch.bool, device=self.device)
        hits[valid_mask] = (grid_device[rr_global[valid_mask].long(), cc_global[valid_mask].long()] == gridmap.OBSTACLE)

        # Identificar el índice de la primera colisión para cada rayo
        # hits is (N, D), so argmax over dim=1 (distance)
        hit_indices = torch.argmax(hits.to(torch.int32), dim=1)
        
        # Manejar el caso donde un rayo no impacta nada (alcanza el rango máximo)
        no_hits = ~torch.any(hits, dim=1)
        hit_indices[no_hits] = self.max_range - 1

        # Coordenadas locales para pintar la imagen del sensor
        rr_local = center + dr
        cc_local = center + dc

        # Asignar valores a la matriz local basados en los índices de impacto
        for i in range(len(angles)):
            idx = hit_indices[i]
            
            # El rayo viaja por espacio libre antes del impacto
            free_rr = rr_local[i, :idx]
            free_cc = cc_local[i, :idx]
            lidar_image[free_rr, free_cc] = gridmap.FREE
            
            # Registrar el obstáculo si hubo impacto dentro del rango
            if not no_hits[i]:
                lidar_image[rr_local[i, idx], cc_local[i, idx]] = gridmap.OBSTACLE

        return lidar_image

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    
    def scan(self, gridmap, position: np.ndarray|torch.Tensor|Iterable[int], orientation: np.ndarray|torch.Tensor ) -> torch.Tensor:
        position = position if isinstance(position, torch.Tensor) else torch.tensor(position, device=self.device)
        orientation = orientation if isinstance(orientation, torch.Tensor) else torch.tensor(orientation, device=self.device)

        return self._perform_scan(gridmap, position, self.angles_rad+orientation)
 