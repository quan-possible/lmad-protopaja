from abc import abstractmethod, ABC
from dataclasses import dataclass
import math

class Heuristic():
    def __init__(self,goal):
        self.goal = goal
    
    def __call__(self,state):
        return euclidean(self.goal.cursor,state.cursor)


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

