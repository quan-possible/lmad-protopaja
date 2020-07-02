from abc import abstractmethod, ABC

class State(ABC):
    """
    This class describes an interface for state objects.
    
    One state can transition into another state by applying some action. 
    The basic interface simply requires a method 'sucessors' which is supposed
    to return all actions applicable to a state and the resulting successor 
    states.

    The decorator @abstractmethod means that an error is raised if the
    corresponding method is not implemented when the child is instansiated.

    """
    
    @abstractmethod
    def successors(self):
        """
        Get list of (Action, State) pairs describing all applicable actions and
        the associated new state.
        
        Returns
        -------
        list of (Action, State) pairs.
           State is a new instance of same class, Action is some structure 
           determined by the inheriting class.
        """
        pass
