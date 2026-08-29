import os
import copy
import pickle
import pandas as pd
import numpy as np
import scipy as sc

try:
    import cupy as cp
    import cupyx as cpx
    if cp.cuda.is_available():
        np = cp  # Use CuPy if a CUDA-enabled GPU is available
        cpx = sc
        print("Using CuPy (GPU accelerated)")
    else:
        print("Using NumPy (CPU) - CuPy installed but no GPU available")
except ImportError:
    print("Using NumPy (CPU) - CuPy not installed")

from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar, Union
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

from Library.Utility import *
from Library.Wrapper import Wrapper
from Library.Random import *
from Library.Parameters import *
from Library.SkewCalibrationBase import *
from Library.OptionPricerKou2002 import kou_call, kou_put
from Library.OptionPricerBSM1973 import *
from Library.SkewCalibrationKimYi2025 import *
