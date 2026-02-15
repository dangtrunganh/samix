from abc import ABC, abstractmethod


class BasePrototypeBasedLoss(ABC):

    def __init__(self):
        self.points = None

    @abstractmethod
    def set_set_prototypes(self, set_prototypes):
        print(f"Shape of fixed protos = {set_prototypes.shape}")
        self.points = set_prototypes
