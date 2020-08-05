# Importing the folder
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

from dataclasses import dataclass # Use dataclass to create hash, eq, and order.
import itertools # For creating combinations.
# Local imports.
import state 
from action import Action
from distance import euclidean
from depth_distance import Measure
import cv2
import math
import numpy as np

# This is the new values for the thresholded pixels.
# When the pixels where the pavements appear is determined,
# they are assigned to the this value.
new_val = 90

@dataclass(eq=True, order=True, unsafe_hash=True)
class PathState(state.State):
    """
    A state in path-planning map. Used by the A-star algorithm.

    The map consists of rows and columns. The shape of the map depends on the input
    of segmentation. 
    
    Our path may move in one of these directions: N,S,E,W,NW,NE,SW,SE.

    The cost of a move is the real-world distance between two locations.

    Attributes
    ----------
    location : (int,int)
       Location of current state
    processed_img : numpy.ndarray
       The image being processed (in other word, the map)
    Measure : object
       Object provides methods to calculate real-world distance.
    
    See
    ---
    mappdistance.py
       For a couple of simple heuristics.
    astar.py
       For a few examples using astar to solve MAPP problems.
       
    """

    # Attributes for dataclass
    cursor : tuple
   
    def __init__(self, location, processed_img, Measure):
        """
        Create new state.
        
        Parameters
        ----------
        See class attributes above.

        Raises
        ------
        ValueError
           If location does not pass 'on_path' and 'blocked' test
           -> For diagnostics reasons.
        """
        
        self.processed_img = processed_img
        self.Measure = Measure
        
        # Check if the state location is on a pixel belongs to the pavement (on path) 
        # and is not blocked.
        if not on_path(self.processed_img,location,new_val) \
                                    and self.Measure.blocked(location):
            raise ValueError(f" NOOOOOOO!!! ")
        self.cursor = location


    def apply(self,action):
        return PathState(action.target, self.processed_img,self.Measure)


    def successors(self):
        """
        Get all possible successor states and associated actions.

        Returns
        -------
        list of (Action, PathState)
           List of actions and new state.
        """
        # This list will store possible target locations.
        moves = []
        
        step = 20

        # Go over all locations.
        for (dr,dc) in ((0,0),(step,0),(-step,0),(0,-step),(0,step),
                        (-step,-step),(-step,step)):
            # Calculate new location.
            (rt,ct) = (self.cursor[0]+dr,self.cursor[1]+dc)
            # Check if new location is on path and not blocked.
            if on_path(self.processed_img,(rt,ct),new_val) and \
               not self.Measure.blocked((rt,ct)):
               moves.append((rt,ct))
        
        # Append the found locations to the list of successors.
        succ = []
        for location in moves:
            cost = self.Measure.measure(self.cursor,location)
            succ.append((Action(self.cursor,location,cost),
                        PathState(location,self.processed_img,self.Measure)))
        return succ

def on_path(image,point,new_val):
   """
   Determine if the pixel belongs to the pavement.

   Helper function

   Parameters
   ----------
   image : numpy.ndarray
   point : (int,int)
   new_val : int
      Brightness value of the pavement.

   Returns
   -------
   Boolean value
   """

   if (point[0] in range(0,image.shape[0])) and (point[1] in range(0,image.shape[1])):
      return image[point[0],point[1]] == new_val
   else:
      return False