from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass