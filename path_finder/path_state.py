# Importing the folder
import state
import numpy as np
import math
import cv2
import pyrealsense2
from action import Action
import itertools  # For creating combinations.
# Use dataclass to create hash, eq, and order.
from dataclasses import dataclass
from distance import euclidean
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'path_finder')

# Local imports.

# This is the substitute for the road_val_range.
# When the pixels where the roads appear is determined,
# they are assigned to the first value of the tuple.
# Every other pixels are assigned to the latter value.
road_val_range = 90, 0


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
    cursor: tuple

    def __init__(self, location, processed_img, Measure):
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
        self.Measure = Measure
        self.blocked = Measure.blocked
        self.measure = Measure.measure
        if self.blocked(location) or not on_path(processed_image, location, euclidean,
                                                 road_val_range[0]):
            raise ValueError(f" Oh no! ")
        self.cursor = location
      #   print(self.cursor)

    def apply(self, action):
        return PathState(action.target, self.processed_img)

    def successors(self):
        """
        Get all possible successor states and associated actions.

        A successor state is possible if all agents move so that there is no
        direct swap of position between any pair of agents, no agent move into
        a wall location or outside the grid, and not more than one agent is
        located at any one position.

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
        for (dr, dc) in ((0, 0), (step, 0), (-step, 0), (0, -step), (0, step),
                         (-step, -step), (-step, step), (step, -step), (-step, step)):
            # Calculate new location.
            (rt, ct) = (self.cursor[0]+dr, self.cursor[1]+dc)
            # Check if new location is in wall or outside.
            # Do not check move to occupied space just yet.
            if not self.blocked((rt, ct)) and \
               on_path(self.processed_img, (rt, ct), road_val_range[0]):
                moves.append((rt, ct))
        # Now we have a list of possible target locations for the next move.
        # The next thing to do is appending all of them to the list of successors.
        succ = []
        for location in moves:
            cost = self.measure(self.cursor, location)
            succ.append((Action(self.cursor, location, cost),
                         PathState(location, self.processed_img, self.Measure)))
        return succ


def on_path(processed_image, point, road_val):
    if (point[0] in range(0, processed_image.shape[0])) and (point[1] in range(0, processed_image.shape[1])):
      # print('concac')
        return processed_image[point[0], point[1]] == road_val
    else:
        # raise ValueError(f" Fuck you! ")
        return False
