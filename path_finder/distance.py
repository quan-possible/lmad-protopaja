from abc import abstractmethod, ABC
from dataclasses import dataclass
from depth_distance import Measure
import math


class Heuristic():
    ''' 
    Heuristics for A-star.

    Attributes
    ----------
    goal : (int,int)
       Coordinate of the goal.
    measure : function
       Function used to measure the distance between a given point and the goal.

    See
    ---
    paint_path.py
       Where it is first defined
    astar.py
       Where it is used.
    '''

    def __init__(self,goal,measure):
        self.goal = goal
        self.measure = measure
    
    def __call__(self,state):
        dist = self.measure(self.goal.cursor,state.cursor)
        return dist


def euclidean(u,v):
    """ 
    Euclidean distance between two points. 
    
    Helper function.
    
    Parameters
    ----------
    u : (int,int)
    v : (int,int)
    Returns
    -------
    int
       Euclidean distance.
    """
    x1,y1 = u
    x2,y2 = v
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)