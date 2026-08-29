import copy
import numpy as np

from Library.Wrapper import Wrapper
from abc import ABC, abstractmethod
from scipy.special import ndtri
from numpy.typing import NDArray
from numpy.random import MT19937, PCG64, Generator, BitGenerator


class RandomBase(ABC):

    def __init__(self, generator: BitGenerator) -> None:
        self.generator = Generator(generator)

    @abstractmethod
    def __deepcopy__(self, memodict={}) -> 'RandomBase':
        pass

    @abstractmethod
    def get_uniform_rv(self, variates: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def set_seed(self, seed: np.uint64) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_gaussian(self, variates: NDArray[np.float64]) -> None:
        self.get_uniform_rv(variates=variates)
        variates[:] = ndtri(variates)

    def get_poisson(self, variates: NDArray[np.int64], lamb=NDArray[np.float64]) -> None:
        variates[:] = self.generator.poisson(lam=lamb, size=variates.shape)

    def get_aded(
            self,
            variates: NDArray[np.float64],
            n: NDArray[np.int64],
            pprob: NDArray[np.float64],
            eta1: NDArray[np.float64],
            eta2: NDArray[np.float64]
    ) -> None:
        k  = self.generator.binomial(n=n, p=pprob, size=variates.shape)
        r  = self.generator.gamma(shape=k, scale=1 / eta1, size=variates.shape)
        r -= self.generator.gamma(shape=n - k, scale=1 / eta2, size=variates.shape)
        variates[:] = r


class RandomMT19937(RandomBase):

    def __init__(self, seed: np.uint64=None):
        if seed is None:
            seed = np.uint64(1)

        self.seed = seed
        self.generator = MT19937(seed=self.seed)
        super().__init__(self.generator)

    def __deepcopy__(self, memodict={}) -> RandomBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def get_uniform_rv(self, variates: NDArray[np.float64]) -> None:
        variates[:] = self.generator.random(size=variates.shape)

    def reset(self) -> None:
        self.generator = MT19937(seed=self.seed)
        super().__init__(self.generator)

    def set_seed(self, seed: np.uint64) -> None:
        self.seed = seed
        self.generator = MT19937(seed=seed)
        super().__init__(self.generator)

    @property
    def seed(self) -> np.uint64:
        return self._seed

    @seed.setter
    def seed(self, seed: np.uint64) -> None:
        self._seed = np.uint64(seed)


class RandomPCG64(RandomBase):

    def __init__(self, seed: np.int64=None):
        if seed is None:
            seed = np.int64(1)

        self.seed = seed
        self.generator = PCG64(seed=self.seed)
        super().__init__(self.generator)

    def __deepcopy__(self, memodict={}) -> RandomBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def get_uniform_rv(self, variates: NDArray[np.float64]) -> None:
        variates[:] = self.generator.random(size=variates.shape)

    def reset(self) -> None:
        self.generator = PCG64(seed=self.seed)
        super().__init__(self.generator)

    def set_seed(self, seed: np.uint64) -> None:
        self.seed = seed
        self.generator = PCG64(seed=seed)
        super().__init__(self.generator)

    @property
    def seed(self) -> np.uint64:
        return self._seed

    @seed.setter
    def seed(self, seed: np.uint64) -> None:
        self._seed = np.uint64(seed)


class RandomAntiThetic(RandomBase):

    def __init__(self, inner_rng: RandomBase) -> None:
        self.generator = inner_rng
        self.generator.reset()

    def __deepcopy__(self, memodict={}) -> RandomBase:
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def get_uniform_rv(self, variates: NDArray[np.float64]) -> None:

        l, m, n = variates.shape
        m_ = int(m / 2)

        x = np.zeros((l, m_, n))
        self.generator.get_uniform_rv(x)
        mask_even = np.arange(m) % 2 == 0
        variates[:, mask_even, :]  = x
        variates[:, ~mask_even, :] = 1. - x

    def reset(self) -> None:
        self.generator.reset()

    def set_seed(self, seed: np.uint64) -> None:
        self.generator.set_seed(seed)

    @property
    def generator(self) -> RandomBase:
        return self._generator

    @generator.setter
    def generator(self, inner_rng: RandomBase) -> None:
        self._generator = Wrapper(inner_rng)
