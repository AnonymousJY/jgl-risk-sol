import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.optimize import newton
# from Library.RootFinder import newton_raphson
from Library.OptionPricerBSM1973 import BlackScholesMertonCall, BlackScholesMertonPut
from Library.OptionPricerHeston1993 import heston_call, heston_put


class SkewCalibrationBase(ABC):

    @abstractmethod
    def target(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def model_vol(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        pass
