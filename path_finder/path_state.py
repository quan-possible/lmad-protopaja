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
import pyrealsense2
import cv2
import math
import numpy as np

# This is the substitute for the road_val_range.
# When the pixels where the roads appear is determined,
# they are assigned to the first value of the tuple.
# Every other pixels are assigned to the latter value.
reclassifying_val = 90,0



@dataclass(eq=True, order=True, unsafe_hash=True)
class PathState(state.State):
    """
    This is Path Planning system for a robot given a picture.

    The picture is a matrix consisting of rows and columns. The coordinates are
    determined as road or not road based on its brightness value. (90 is road)
    
    The path may move N,NW,NE,S,E,W in the picture (but not to anywhere that is not the road).

    The cost of a move is dependent on the direction


    Attributes
    ----------
    processed_img : ndarray
       The image already processed (coverted to grayscale and )
    ncols : int > 0
       Number of columns in grid.
    agents : tuple of (int,int) pairs
       Location of all agents (list index i is the location of agent i).
    walls : set of (int,int)
       Location of walls (cells that cannot be entered by an agent).

    Note
    ----
    Ordering and performance : This class is supposed to be used with the A-star
    algorithm, which is based around `queue.PriorityQueue`. PriorityQueue 
    primarily orders its items based on the priority score (lowest first), but
    in the event that two scores are the same it requires the data object to 
    have an order (a __lt__ method). In the case when there are multiple states
    with the same score the running time can be significantly impacted by the
    state to state ordering. This class simply uses `dataclass` to automatically
    create an ordering, which means that there are some configurations 
    (especially with open spaces) when finding a path in one direction (S->G) is
    much slower/faster than the other (G->S). Some specific implementations of
    __lt__ may give a better on-average performance.
    
    See
    ---
    mappdistance.py
       For a couple of simple heuristics.
    astar.py
       For a few examples using astar to solve MAPP problems.
    """

    # Attributes for dataclass
    cursor : tuple
   
    def __init__(self, location, processed_img):
        """
        Create new state.
        
        Parameters
        ----------
        See class attributes above.

        Raises
        ------
        ValueError
           If agent locations are on walls, outside the grid, or not unique.
        """
        self.processed_img = processed_img
        if not blocked(self.processed_img,location,reclassifying_val[0]):
            raise ValueError(f" Not on the road! ")
        self.cursor = location
      #   print(self.cursor)


    def apply(self,action):
        return PathState(action.target, self.processed_img)


    def successors(self):
        """
        Get all possible successor states and associated actions.

        A successor state is possible if all agents move so that there is no
        direct swap of position between any pair of agents, no agent move into
        a wall location or outside the grid, and not more than one agent is
        located at any one position.

        Theory
        ------
        One can define simultaneous moves by agents different ways;
        
        1. The strictest definition requires that every agent is moving to
        an empty cell. Transition from ...12... to ....12.. is not allowed.
        
        2. A looser definition does not allow any cycles in the moves.
        Transition from ...12... to ....12.. is OK
        but transition from ...12... to ...21... is not.

        The definition used here forbids cycles involving 2 agents. But longer cycles are OK,
        for example
        from ..12.. to ..31..
             ..34..    ..42..
        as this does not involve agents jumping over each other.

        Agents may also stand still.

        Our agents move to the 4 cardinal directions N, S, W and E only.
        One could of course allow also the intermediate NE, NW, SE, SW.
        

        Returns
        -------
        list of (Action, MAPPGridState)
           List of actions and new state.
        """
        # This list will store possible target locations for each agent.
        moves = []
        step = 30
        # Go over all agents/locations.
        # The index is the agent id so simply iterate over it.
        # for (r,c) in self.agents:
        #     # Add possible Manhattan moves (north, south, east, west) and no
        #     # move as a list.
        #     moves.append([])
      #   for (dr,dc) in ((0,0),(step,0),(-step,0),(0,-step),(0,step)):
        for (dr,dc) in ((0,0),(step,0),(-step,0),(0,-step),(0,step),
                        (-step,-step),(-step,step),(step,-step), (-step,step)):
            # Calculate new location.
            (rt,ct) = (self.cursor[0]+dr,self.cursor[1]+dc)
            # Check if new location is in wall or outside.
            # Do not check move to occupied space just yet.
            if blocked(self.processed_img,(rt,ct),reclassifying_val[0]):
               moves.append((rt,ct))
        # Now we have a list of possible target locations for the next move.
        # The next thing to do is appending all of them to the list of successors.
        succ = []
        for location in moves:
            cost = euclidean(self.cursor,location)
            succ.append((Action(self.cursor,location,cost),
                        PathState(location, processed_img = self.processed_img)))
        return succ

""" 
Determine if the given point is outside the road.

Helper function.

Parameters
----------
image: numpy array
   Reference Image
given_point: pair of float
reclassifying_val: pair of int
   Road value

Returns
-------
Boolean

"""
def blocked(image,given_point,ref_val):
   if (given_point[0] in range(0,image.shape[0])) and (given_point[1] in range(0,image.shape[1])):
      return image[given_point[0],given_point[1]] == ref_val
   else:
      return False
