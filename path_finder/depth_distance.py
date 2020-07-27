import pyrealsense2 as rs
from abc import abstractmethod, ABC
from dataclasses import dataclass
import math
import pyrealsense2 as rs


class Measure:

    def __init__(self, depth_frame, color_frame, depth_scale, obstacles):
        self.depth_frame = depth_frame
        self.color_frame = color_frame
        self.depth_scale = depth_scale
        self.obstacles = obstacles
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

    def measure(self, begin, end):
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

    def blocked(self, point, min_dist=0):
        """
        Determine if the given point is outside the road.

        Helper function.

        Parameters
        ----------
        image: numpy array
        Reference Image
        point: pair of float
        reclassifying_val: pair of int
        Road value

        Returns
        -------
        Boolean

        """
        is_blocked = False
        i = 0
        if self.obstacles:
            while not is_blocked and i < len(self.obstacles):
                if self.obstacles[i]:
                    obstacle = self.obstacles[i][0]
                    if self.measure(point, obstacle) < min_dist:
                        is_blocked = True
                
                i += 1

        return is_blocked
