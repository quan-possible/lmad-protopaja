from abc import abstractmethod, ABC
from dataclasses import dataclass
from depth_distance import Measure
import math

class Heuristic():
    def __init__(self,goal,measure):
        self.goal = goal
        self.measure = measure
    
    def __call__(self,state):
        dist = self.measure(self.goal.cursor,state.cursor)
        return dist


def euclidean(u,v):
    """ 
    Manhattan distance between two points. 
    
    Helper function.
    
    Parameters
    ----------
    u : pair of (int,int)
    v : pair of (int,int)
    Returns
    -------
    int
       Manhattan distance.
    """
    x1,y1 = u
    x2,y2 = v
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)