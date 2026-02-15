from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseLoss(ABC):
    @staticmethod
    @abstractmethod
    def add_specific_args(parser: ArgumentParser):
        """Add model specific args here"""
        pass
