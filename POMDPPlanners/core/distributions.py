from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass
    
    def probability(self, value):
        raise NotImplementedError("The method is not implemented for this distribution.")
    
class DiscreteDistribution(Distribution):
    def __init__(self, values: list, probs: np.array):
        assert len(values) == len(probs)
        assert sum(probs) == 1
        
        self.values = values
        self.probs = probs
        
    def sample(self):
        return np.random.choice(self.values, p=self.probs)
    
    def probability(self, value):
        return self.probs[self.values.index(value)]
        
class Numpy2DDistribution(Distribution):
    def __init__(self, values: np.ndarray, probs: np.array):
        assert values.shape[0] == 2
        assert values.shape[1] == len(probs)
        
        self.values = values
        self.probs = probs

    def sample(self):
        idx = np.random.choice(len(self.probs), p=self.probs)
        return self.values[:, idx]
