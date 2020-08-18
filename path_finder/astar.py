# Append 'path_finder' folder to the directory
import sys
sys.path.insert(1, 'path_finder')
# Basic imports
from math import inf
# Local imports
from queue import PriorityQueue
from path_state import *
from distance import Heuristic
import action


def astar(start_state, goaltest, h):
    """
    Perform A-star search.

    Finds a sequence of actions from `start_state` to some end state satisfying 
    the `goaltest` function by performing A-star search.

    This function returns a policy, i.e. a sequence of actions which, if
    successively applied to `start_state` will transform it into a state which
    satisfies `goaltest`.

    Parameters
    ----------
    start_state : State
       State object with `successors` function.
    goaltest : Function (State -> bool)
       A function which takes a State object as parameter and returns True if 
       the state is an acceptable goal state.
    h : Function (State -> float)
       Heuristic function estimating the distance from a state to the goal.
       This is the h(s) in f(s) = h(s) + g(s).
    
    Returns
    -------
    list of actions
       The policy for transforming start_state into one which is accepted by
       `goaltest`.
    """
    # Dictionary to look up predecessor states and the
    # the actions which took us there.
    predecessor = {}

    # Dictionary holding the (yet) best found distance to a state,
    # the function g(s) in the formula f(s) = h(s) + g(s).
    g = {}

    # Priority queue holding states to check, the priority of each state is
    # f(s).
    # Elements are encoded as pairs of (prio, state)
    Q = PriorityQueue()

    Q.put((h(start_state), start_state))
    min_g = inf
    goal = start_state
    g[start_state] = 0
    if goaltest(start_state):
        return []

    while not Q.empty():
        # Pop the state with biggest f value.
        f, state = Q.get()
        if f < min_g:
            for (action, ss) in state.successors():
                # Append successor states that we have not discovered yet,
                # or ones with existing g value higher than what we
                # just found.
                if ss not in g.keys() or g[ss] > (g[state] + action.cost):
                    g[ss] = g[state] + action.cost
                    predecessor[ss] = (state, action)
                    Q.put((g[ss] + 2*h(ss), ss)) # Weighted for better execution speed 

                if goaltest(ss):
                    min_g = g[ss]
                    goal = ss

        # If there is no other state with f value smaller than the
        # best g value we have found (i.e. no potentially shorter path),
        # end the search.
        else:
            last_state, last_action = predecessor[goal]
            pi = [(last_action.source, last_action.target)]

            while last_state != start_state:
                (last_state, last_action) = predecessor[last_state]
                pi.append((last_action.source, last_action.target))

            if len(pi) != 0:
                return reversed(pi)
            else:
                return None

    return None

