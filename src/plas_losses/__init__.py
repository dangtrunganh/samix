from src.plas_losses.main_loss_fnc2 import FNC2
from src.plas_losses.main_loss_dot_regression import DotRegression
from src.plas_losses.loss_supcon import SupConLoss
from src.plas_losses.main_loss_nc_samix import NCSAMix

__all__ = [
    "FNC2",
    "SupConLoss",
    "NCSAMix",
    "DotRegression"
]

LIST_LOSSES_PROTOTYPES = [
    "fnc2",
    "dot_regression",
    "nc_samix"
]

LOSSES = {
    "fnc2": FNC2,
    "supcon": SupConLoss,
    "nc_samix": NCSAMix,
    "dr": DotRegression
}
