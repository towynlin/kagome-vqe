import numpy as np
from qiskit.algorithms.optimizers.optimizer import (
    Optimizer,
    OptimizerResult,
    OptimizerSupportLevel,
    POINT,
)
from typing import Callable, List, Optional, Tuple, Union


HALF_PI = np.pi / 2


class Rotosolve(Optimizer):
    """Rotosolve gradient-free optimizer"""

    def __init__(self, maxiter: int = 100):
        """
        Initialize the optimization algorithm, setting the support
        level for _gradient_support_level, _bound_support_level,
        _initial_point_support_level, and empty options.
        """
        self._gradient_support_level = self.get_support_level()["gradient"]
        self._bounds_support_level = self.get_support_level()["bounds"]
        self._initial_point_support_level = self.get_support_level()["initial_point"]
        self._options = {}
        self._max_evals_grouped = 1
        self.maxiter = maxiter

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": OptimizerSupportLevel.ignored,
            "bounds": OptimizerSupportLevel.ignored,
            "initial_point": OptimizerSupportLevel.required,
        }

    def minimize(
        self,
        fun: Callable[[POINT], Union[float, List[float]]],
        x0: POINT,
        jac: Optional[Callable[[POINT], POINT]] = None,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> OptimizerResult:
        """Minimize the scalar function.

        Args:
            fun: The scalar function to minimize.
            x0: The initial point for the minimization.
            jac: The gradient of the scalar function ``fun``.
            bounds: Bounds for the variables of ``fun``. This argument might be ignored if the
                optimizer does not support bounds.

        Returns:
            The result of the optimization, containing e.g. the result as attribute ``x``.
        """
        D = len(x0)  # type: ignore
        𝜃 = np.tile(x0, 4)
        for _ in range(self.maxiter):
            for d in range(D):
                𝜃[d] = 0
                d_plus = d + D
                𝜃[d_plus] = HALF_PI
                d_minus = d_plus + D
                𝜃[d_minus] = -HALF_PI

                # Run the quantum circuit job
                energies = fun(𝜃)

                a = np.arctan2(
                    2 * energies[0] - energies[1] - energies[2],  # type: ignore
                    energies[1] - energies[2],  # type: ignore
                )
                𝜃[d] = -HALF_PI - a
                if 𝜃[d] <= -np.pi:
                    𝜃[d] += 2 * np.pi
                𝜃[d_plus] = 𝜃[d]
                𝜃[d_minus] = 𝜃[d]
                𝜃[d_minus + D] = 𝜃[d]

        final_point = 𝜃[:D]
        result = OptimizerResult()
        result.x = final_point
        result.fun = fun(final_point)  # type: ignore
        return result
