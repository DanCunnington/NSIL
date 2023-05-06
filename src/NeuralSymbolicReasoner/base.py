from abc import ABC, abstractmethod


class NeuralSymbolicReasoner(ABC):
    def __init__(self, args, logger, task):
        self.args = args
        self.logger = logger
        self.task = task

    @abstractmethod
    def train(self, h):
        """
        Train the neural network using neural-symbolic reasoning
        @param h: the converted hypothesis
        """
        pass




