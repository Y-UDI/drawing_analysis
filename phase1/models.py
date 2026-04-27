from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PreprocessResult:
    original:    np.ndarray
    denoised:    np.ndarray
    binarized:   np.ndarray
    deskewed:    np.ndarray
    sr_image:    Optional[np.ndarray]
    skew_angle:  float
    quality_in:  dict
    quality_out: dict
    elapsed_sec: float
