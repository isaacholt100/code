from typing import Callable
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]
ComplexArray = npt.NDArray[np.complexfloating | np.floating]
Curves = Callable[..., list[float]]