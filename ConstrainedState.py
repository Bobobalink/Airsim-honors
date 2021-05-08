from dataclasses import dataclass
import numpy as np


@dataclass
class ConstrainedState:
    pos: np.ndarray
    accumPos: float
    curvature: float  # curvature of path at point (radians / meter)
    velocity: float  # maximum velocity at this point
    minACC: float  # min acceleration (negative)
    maxACC: float  # max acceleration (positive)
