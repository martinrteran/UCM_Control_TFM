import numpy as np
from typing import Iterable, Tuple, Union
import torch
from tfm_control_ucm.core.grid_env.map import GridMap

class GridLidar:
    """
    A 2D LIDAR sensor that casts multiple rays in a configurable field of view.
    Base class providing raycasting computation (cached or on-the-fly).
    """
    def __init__(
        self,
        *,
        num_rays: int = 10,
        max_range: int = 10,
        fov: float = np.pi,
        noise_std: float = 0.001,
        with_cache: bool = True,
        device: Union[torch.device, str] = "cpu",
        chunk_size: int = 5000
    ):
        if not isinstance(num_rays, (np.integer, int)): raise TypeError("The num_rays must be an integer")
        if not isinstance(max_range, (np.integer, int)): raise TypeError("The max_range must be an integer")
        if not isinstance(fov, (np.floating, float)): raise TypeError("The fov must be a float")
        if not isinstance(noise_std, (np.floating, float)): raise TypeError("The noise_std must be a float")
        if not isinstance(with_cache, bool): raise TypeError("The with_cache must be a boolean")
        if not isinstance(device, (str, torch.device)): raise TypeError("The device must be a torch.device")

        if num_rays <= 0: raise ValueError("The num_rays must be greater than zero")
        if max_range <= 0: raise ValueError("The max_range must be greater than zero")
        if fov <= 0: raise ValueError("The fov must be greater than zero")
        if noise_std < 0: raise ValueError("The noise_std must be zero or positive")

        self.num_rays = num_rays
        self.max_range = max_range
        self.fov_rad = fov
        self.noise_std = noise_std
        self.with_cache = with_cache
        self.device = torch.device(device) if isinstance(device, str) else device
        
        angles = torch.linspace(-self.fov_rad/2, self.fov_rad/2, self.num_rays, device=self.device)
        self.angles_rad = angles.unsqueeze(1)

        if self.with_cache:
            self.dist_grid = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
            self.dr_base = torch.round(torch.sin(self.angles_rad) * self.dist_grid).to(torch.int32)
            self.dc_base = torch.round(torch.cos(self.angles_rad) * self.dist_grid).to(torch.int32)

        self._map_caches = {}
        self.chunk_size = chunk_size
    
    def get_config(self) -> dict:
        return {
            'num_rays': self.num_rays, 'max_range': self.max_range, 
            'fov': self.fov_rad, 'noise_std': self.noise_std, 
            'with_cache': self.with_cache, 'device': str(self.device)
        }
    
    @classmethod
    def config_keys(cls) -> list:
        return ['num_rays', 'max_range', 'fov', 'noise_std', 'with_cache', 'device']

    def _build_full_map_cache(self, gridmap: GridMap) -> torch.Tensor:
        height, width = gridmap.grid.shape
        cache_table = torch.full((height, width, self.num_rays), self.max_range, dtype=torch.float32, device=self.device)
        grid_device = gridmap.grid.to(self.device)
        
        free_rows, free_cols = torch.where(grid_device != gridmap.OBSTACLE)
        if len(free_rows) == 0:
            return cache_table

        dist = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
        dr_base = torch.round(torch.sin(self.angles_rad) * dist).to(torch.int32)
        dc_base = torch.round(torch.cos(self.angles_rad) * dist).to(torch.int32)

        chunk_size = self.chunk_size
        for i in range(0, len(free_rows), chunk_size):
            r_chunk = free_rows[i : i + chunk_size]
            c_chunk = free_cols[i : i + chunk_size]
            
            cc_global = c_chunk[:, None, None] + dc_base[None, :, :]
            rr_global = r_chunk[:, None, None] - dr_base[None, :, :]
            
            valid_mask = (rr_global >= 0) & (rr_global < height) & (cc_global >= 0) & (cc_global < width)
            rr_clamped = torch.clip(rr_global, 0, height - 1)
            cc_clamped = torch.clip(cc_global, 0, width - 1)
            
            map_hits = grid_device[rr_clamped, cc_clamped] == gridmap.OBSTACLE
            
            hits = torch.ones_like(rr_global, dtype=torch.bool)
            hits[valid_mask] = map_hits[valid_mask]
            
            hit_indices = torch.argmax(hits.to(torch.int32), dim=2)
            no_hits = ~torch.any(hits, dim=2)
            
            dists = (hit_indices + 1).to(torch.float32)
            dists[no_hits] = float(self.max_range)
            
            cache_table[r_chunk, c_chunk] = dists
            
        return cache_table

    def _is_full_circle(self) -> bool:
        return abs(self.fov_rad - 2 * np.pi) < 1e-6

    def _compute_distances(self, gridmap: GridMap, position: torch.Tensor, heading: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula las colisiones de los rayos delegando en caché o procesamiento on-the-fly.
        Devuelve (hit_distances, angles), con los ángulos relativos al rumbo del robot.

        La caché guarda los rayos en el marco del mundo. Con un campo de visión de 360°
        girar el robot solo cambia el ángulo relativo de cada rayo; con un campo menor
        hay que lanzar los rayos girados, así que se calculan on-the-fly.
        """
        heading_is_zero = bool(torch.all(heading == 0))
        if self.with_cache and (heading_is_zero or self._is_full_circle()):
            if gridmap.name not in self._map_caches:
                self._map_caches[gridmap.name] = self._build_full_map_cache(gridmap)

            c, r = position.long()
            hit_distances = self._map_caches[gridmap.name][r, c].clone()
            world = self.angles_rad.flatten()
            angles = torch.atan2(torch.sin(world - heading), torch.cos(world - heading))

        else:
            dist = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
            world = (self.angles_rad + heading).flatten()
            angles = self.angles_rad.flatten().clone()

            dr = torch.round(-torch.sin(world)[:, None] * dist).to(torch.int32)
            dc = torch.round(torch.cos(world)[:, None] * dist).to(torch.int32)

            cc_global = position[0].long() + dc
            rr_global = position[1].long() + dr

            height, width = gridmap.grid.shape
            valid_mask = (rr_global >= 0) & (rr_global < height) & (cc_global >= 0) & (cc_global < width)
            
            grid_device = gridmap.grid.to(self.device)
            hits = torch.ones_like(rr_global, dtype=torch.bool)
            hits[valid_mask] = (grid_device[rr_global[valid_mask], cc_global[valid_mask]] == gridmap.OBSTACLE)

            hit_indices = torch.argmax(hits.to(torch.int32), dim=1)
            no_hits = ~torch.any(hits, dim=1)
            
            hit_distances = (hit_indices + 1).to(torch.float32)
            hit_distances[no_hits] = float(self.max_range)

        if self.noise_std > 0:
            noise = torch.normal(0, self.noise_std, size=hit_distances.shape, device=self.device)
            noise[hit_distances == float(self.max_range)] = 0.0  # Sin ruido si no impacta
            hit_distances = torch.clip(hit_distances + noise, 0.0, float(self.max_range))

        return hit_distances, angles

    def scan(self, gridmap: GridMap, position: Union[np.ndarray, torch.Tensor, Iterable[int]], heading: Union[float, np.ndarray, torch.Tensor] = 0.0) -> torch.Tensor:
        """
        Cast the rays from a cell.

        Parameters
        ----------
        gridmap : GridMap
            Map to scan.
        position : array-like
            Robot position (col, row).
        heading : float, optional
            Direction the robot faces (radians, 0 = east, counter-clockwise).

        Returns
        -------
        torch.Tensor
            (num_rays, 2) rows of [distance, angle relative to the heading].
        """
        position = position if isinstance(position, torch.Tensor) else torch.tensor(np.asarray(position), device=self.device, dtype=torch.float32)
        heading = heading if isinstance(heading, torch.Tensor) else torch.tensor(float(np.asarray(heading)), device=self.device, dtype=torch.float32)

        hit_distances, angles = self._compute_distances(gridmap, position, heading)
        return torch.column_stack((hit_distances, angles))

# class GridLidar_2D(GridLidar):
#     """
#     A 2D LIDAR sensor that returns a local occupancy grid image.
#     Implements a full 2D map caching system for O(1) retrieval when noise_std == 0.
#     """
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._image_caches = {}
        
#         # Precalcular el sistema de coordenadas locales en la inicialización
#         # Esto elimina los cálculos trigonométricos (sin, cos) durante el escaneo
#         self.grid_size = 2 * self.max_range + 1
#         self.center = self.max_range
        
#         self.dist_grid_local = torch.arange(1, self.max_range + 1, device=self.device, dtype=torch.float32)
#         local_angles = self.angles_rad.flatten()
        
#         dr_local = torch.round(-torch.sin(local_angles)[:, None] * self.dist_grid_local).to(torch.long)
#         dc_local = torch.round(torch.cos(local_angles)[:, None] * self.dist_grid_local).to(torch.long)
        
#         self.rr_local = self.center + dr_local
#         self.cc_local = self.center + dc_local

#     def _build_full_image_cache(self, gridmap: GridMap) -> torch.Tensor:
#         """
#         Precalcula la imagen del Lidar para TODAS las celdas libres del mapa.
#         Retorna un tensor [H, W, grid_size, grid_size].
#         """
#         # 1. Aseguramos tener las distancias vectorizadas del padre
#         if gridmap.name not in self._map_caches:
#             self._map_caches[gridmap.name] = self._build_full_map_cache(gridmap)
            
#         distances = self._map_caches[gridmap.name]
#         height, width = gridmap.grid.shape
        
#         # 2. Inicializar caché masivo en VRAM
#         image_cache = torch.full((height, width, self.grid_size, self.grid_size), -1, dtype=torch.int8, device=self.device)
        
#         grid_device = gridmap.grid.to(self.device)
#         free_rows, free_cols = torch.where(grid_device != gridmap.OBSTACLE)
        
#         if len(free_rows) == 0:
#             return image_cache
            
#         # El centro siempre está libre si el agente está en una celda válida
#         image_cache[free_rows, free_cols, self.center, self.center] = 0
        
#         # 3. Procesamiento por lotes (chunks) para evitar saturar la VRAM
#         chunk_size = 5000
#         for i in range(0, len(free_rows), chunk_size):
#             r_chunk = free_rows[i : i + chunk_size]
#             c_chunk = free_cols[i : i + chunk_size]
            
#             # Extraer distancias del chunk: [C, N] -> [C, N, 1]
#             hd_discrete = torch.round(distances[r_chunk, c_chunk]).long().unsqueeze(2)
            
#             is_hit = hd_discrete < self.max_range
#             d_grid = self.dist_grid_local.view(1, 1, -1) # [1, 1, D]
            
#             free_mask = (d_grid < hd_discrete) | ((d_grid == self.max_range) & ~is_hit)
#             obs_mask = (d_grid == hd_discrete) & is_hit
            
#             # Expandir coordenadas locales para el lote actual
#             rr_c = self.rr_local.unsqueeze(0).expand(len(r_chunk), -1, -1)
#             cc_c = self.cc_local.unsqueeze(0).expand(len(r_chunk), -1, -1)
#             batch_idx = torch.arange(len(r_chunk), device=self.device).view(-1, 1, 1).expand_as(rr_c)
            
#             img_batch = torch.full((len(r_chunk), self.grid_size, self.grid_size), -1, dtype=torch.int8, device=self.device)
#             img_batch[:, self.center, self.center] = 0
            
#             # Asignación masiva paralela
#             img_batch[batch_idx[free_mask], rr_c[free_mask], cc_c[free_mask]] = gridmap.FREE
#             img_batch[batch_idx[obs_mask], rr_c[obs_mask], cc_c[obs_mask]] = gridmap.OBSTACLE
            
#             image_cache[r_chunk, c_chunk] = img_batch
            
#         return image_cache

#     def scan(self, gridmap: GridMap, position: Union[np.ndarray, torch.Tensor, Iterable[int]], orientation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
#         position = position if isinstance(position, torch.Tensor) else torch.tensor(position, device=self.device, dtype=torch.float32)
#         orientation = orientation if isinstance(orientation, torch.Tensor) else torch.tensor(orientation, device=self.device, dtype=torch.float32)

#         # ESTRATEGIA A: Extracción O(1) de la matriz final.
#         # Solo válido si hay caché activo y no hay ruido, ya que el ruido cambia por frame.
#         if self.with_cache and self.noise_std == 0:
#             if gridmap.name not in self._image_caches:
#                 self._image_caches[gridmap.name] = self._build_full_image_cache(gridmap)
            
#             c, r = position.long()
#             return self._image_caches[gridmap.name][r, c].clone()

#         # ESTRATEGIA B: Reconstrucción ultra rápida.
#         # Usado si with_cache=False, o si noise_std > 0 (requiere recalcular el mapeo del ruido).
#         # Extrae las distancias del caché del padre (que ya incluyen ruido si procede).
#         hit_distances, _ = self._compute_distances(gridmap, position, orientation)
        
#         lidar_image = torch.full((self.grid_size, self.grid_size), -1, dtype=torch.int8, device=self.device)
#         lidar_image[self.center, self.center] = 0

#         hd_discrete = torch.round(hit_distances).long().unsqueeze(1)
#         is_hit = hd_discrete < self.max_range
        
#         free_mask = (self.dist_grid_local < hd_discrete) | ((self.dist_grid_local == self.max_range) & ~is_hit)
#         obs_mask = (self.dist_grid_local == hd_discrete) & is_hit

#         # Reconstrucción sin trig utilizando las coordenadas cacheadas
#         lidar_image[self.rr_local[free_mask], self.cc_local[free_mask]] = gridmap.FREE
#         lidar_image[self.rr_local[obs_mask], self.cc_local[obs_mask]] = gridmap.OBSTACLE

#         return lidar_image

