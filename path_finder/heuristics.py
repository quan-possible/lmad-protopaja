from abc import abstractmethod, ABC
from dataclasses import dataclass
import math
import pyrealsense2 as rs

class Heuristic():
    def __init__(self,goal,measure):
        self.goal = goal
        self.measure = measure 
    
    def __call__(self,state):
        return measure(self.goal.cursor,state.cursor)


def euclidean(u,v):
    """ 
    Euclidean distance between two points. 
    
    Helper function.
    
    Parameters
    ----------
    u : pair of (int,int)
    v : pair of (int,int)

    Returns
    -------
    int
       Euclidean distance.
    """
    x1,y1 = u
    x2,y2 = v
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)



def calculate_distance(color_intrin,beginning,target):
    color_intrin = self.color_intrin
    ix,iy = self.ix, self.iy
    udist = self.depth_frame.get_distance(ix,iy)
    vdist = self.depth_frame.get_distance(x, y)
    #print udist,vdist

    point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [ix, iy], udist)
    point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x, y], vdist)
    #print str(point1)+str(point2)

    dist = math.sqrt(
        math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1],2) + math.pow(
            point1[2] - point2[2], 2))
    #print 'distance: '+ str(dist)
    return dist

