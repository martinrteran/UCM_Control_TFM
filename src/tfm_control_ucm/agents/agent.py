"""
simple_agent.py

A simple reactive agent that uses LIDAR and robot orientation
to choose actions intelligently.

Actions:
    0 = forward
    1 = turn left
    2 = turn right

Author: Martin
"""

import numpy as np


class SimpleAgent:
    def __init__(self, *, turn_threshold=1.5):
        """
        Parameters
        ----------
        turn_threshold : float
            If the front LIDAR rays detect an obstacle closer than this
            distance, the agent will turn instead of moving forward.
        """
        self.turn_threshold = turn_threshold

    def act(self, observation: np.ndarray, orientation: str) -> int:
        """
        Choose an action based on LIDAR distances and robot orientation.

        Parameters
        ----------
        observation : np.ndarray
            LIDAR distances (num_rays,)
        orientation : str
            One of: "N", "E", "S", "W"

        Returns
        -------
        action : int
            0 = forward
            1 = turn left
            2 = turn right
        """

        num_rays = len(observation)
        center = num_rays // 2

        # Split LIDAR into left / center / right sectors
        left_sector = observation[:center]
        right_sector = observation[center + 1 :]
        front_sector = observation[center - 1 : center + 2]

        min_front = np.min(front_sector)

        # If obstacle too close → turn toward the side with more space
        if min_front < self.turn_threshold:
            left_space = np.mean(left_sector)
            right_space = np.mean(right_sector)

            if left_space > right_space:
                return 1  # turn left
            else:
                return 2  # turn right

        # Otherwise → go forward
        return 0