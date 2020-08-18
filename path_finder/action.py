# Basic imports
from dataclasses import dataclass
from typing import Any

# This is a dataclass, we want equality and repr created automatically.
@dataclass(eq=True,repr=True)
class Action:
    """
    Describes an action from source to target state at some cost (reward).

    This class is basically syntactic sugar to structure the interface of the
    different search methods (we could have worked with tuples).

    Attributes
    ----------
    source : State (object)
       Start location.
    target : State (object)
       End location.
    cost : float
       Cost associated with action.
    translate : pair of State
       Tuple of Start and End locations.         
    """

    # Dataclass attributes
    source: Any
    target: Any
    cost: float

    def __init__(self, source, target, cost=1):
        """
        Create an action from `source` to `target` with some `cost`.

        Cost parameter is optional.
        """
        self.source = source
        self.target = target
        self.cost = cost
