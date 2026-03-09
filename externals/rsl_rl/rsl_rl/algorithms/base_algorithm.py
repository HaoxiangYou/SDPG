from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def test_mode(self):
        """Set the algorithm to evaluation mode."""
        pass

    @abstractmethod
    def train_mode(self):
        """Set the algorithm to training mode."""
        pass

    @abstractmethod
    def act(self):
        """Output actions based on observations."""
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def process_env_step(self):
        """Process the environment step, including rewards and done signals."""
        pass

    @abstractmethod
    def compute_returns(self):
        """Compute the returns for the collected experiences."""
        pass
