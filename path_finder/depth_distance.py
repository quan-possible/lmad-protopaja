# Basic imports
from abc import abstractmethod, ABC
from dataclasses import dataclass
import math
import pyrealsense2 as rs


class Measure:
    """
    Object utilizing data from the Realsense camera to
    carry out depth-related operations.
    
    Attributes
    ----------
    depth_frame: 
        See 'pyrealsense' documentation.
    color_frame:
        See 'pyrealsense' documentation.
    depth_scale: float
        The scale of the stream of depth coming from the Realsense camera.
        (For example, depth_scale=0.001 means a pixel value of 1000 equals
         1 meter in real life)
    obstacles: list of list of (int,int)
        coordinates of pixels that form the outline of obstacles.
    """

    def __init__(self, depth_frame, color_frame, depth_scale, obstacles):
        self.depth_frame = depth_frame
        self.color_frame = color_frame
        self.depth_scale = depth_scale
        self.obstacles = obstacles
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    def measure(self, begin, end):
        """
        Measure distance between 2 points.

        Parameters
        ----------
        begin : (int,int)
        end : (int,int)

        Returns
        -------
        Float
            Distance between the 2 points begin and end.
        """

        color_intrin = self.color_intrin
        iy, ix = begin
        y, x = end
        udist = self.depth_frame.get_distance(ix, iy)
        vdist = self.depth_frame.get_distance(x, y)

        point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
        point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
        # print str(point1)+str(point2)

        dist = math.sqrt(
            math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
                point1[2] - point2[2], 2))
        return dist

    def blocked(self, point, min_dist=0.2):
        """
        Determine if the given point is blocked by any obstacle.

        Parameters
        ----------
        point : (int,int)
        min_dist : float
            Maximum distance to the obstacles for the point to be considered blocked.

        Returns
        -------
        Boolean
            True if blocked, False otherwise.
        """

        is_blocked = False
        i = 0

        if self.obstacles:
            while not is_blocked and i < len(self.obstacles):
                if self.obstacles[i]:
                    obstacle = self.obstacles[i][0]
                    dis = self.measure(point, obstacle)
                    if dis < min_dist:
                        is_blocked = True
                        break

                i += 1

        return is_blocked
