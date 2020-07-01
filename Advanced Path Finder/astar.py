import math
from queue import PriorityQueue
from path_state import PathState
from distance import Heuristic
import action
from math import inf


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
    # the actions which took us there. It is empty to start with.

    predecessor = {}

    # Dictionary holding the (yet) best found distance to a state,
    # the function g(s) in the formula f(s) = h(s) + g(s).

    g = {}

    # Priority queue holding states to check, the priority of each state is
    # f(s).
    # Elements are encoded as pairs of (prio, state),
    # e.g. Q.put( (prio, state ))
    # And gotten as (prio,state) = Q.get()

    Q = PriorityQueue()

    # TASK
    # ---------------------------------
    # Complete the A* star implementation.
    # Some variables have already been declared above (others may be needed
    # depending on your implementation).
    # Remember to return the plan (list of Actions).
    #
    # You can look at bfs.py to see how a compatible BFS algorithm can be
    # implemented.
    #
    # The A* algorithm can be found in the MyCourses material.
    #
    # Take care that you don't implement the GBFS algorithm by mistake:
    #  note that you should return a solution only when you *know* it is
    #  optimal (how?)
    #
    # Good luck!

    Q.put((h(start_state), start_state))
    min_g = inf
    goal = start_state
    g[start_state] = 0
    if goaltest(start_state):
        return []

    while not Q.empty():
        f, state = Q.get()
        if f < min_g:
            for (action, ss) in state.successors():
                if ss not in g.keys() or g[ss] > (g[state] + action.cost):
                    g[ss] = g[state] + action.cost
                    predecessor[ss] = (state, action)
                    Q.put((g[ss] + h(ss), ss))
                if goaltest(ss):
                    min_g = g[ss]
                    goal = ss
        else:
            last_state, last_action = predecessor[goal]
            pi = [last_action.translate]

            while last_state != start_state:
                (last_state, last_action) = predecessor[last_state]
                pi.append(last_action.translate)
            return reversed(pi)

    return None

