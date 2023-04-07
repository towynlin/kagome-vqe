from dataclasses import dataclass, field
import numpy as np
from time import strftime
from typing import Any, Tuple


@dataclass
class SimpleRepository:
    parameters: np.ndarray = field()
    gate_names: np.ndarray = field()
    values: np.ndarray = field()

    def __init__(self, num_params: int):
        self.values = np.empty((0,), dtype=float)
        self.parameters = np.empty((0, num_params), dtype=float)
        self.gate_names = np.empty((0, num_params), dtype=str)

    def update(
        self, count: int, parameters: np.ndarray, energy: float, meta: dict[str, Any]
    ):
        self.values = np.append(self.values, energy)
        self.parameters = np.append(self.parameters, [parameters], axis=0)
        t = strftime("%m/%d %H:%M:%S%z")
        print(f"{t} Iteration {count}:\tenergy: {energy: 012.08f}")
        if self.values.size % 24 == 0:
            print(f"\nParameters: {parameters}\n")

    def get_best_result(self) -> Tuple[float, np.ndarray, np.ndarray]:
        if self.values.size > 0:
            return (self.values[-1], self.parameters[-1], np.array([]))
        else:
            return (0, np.array([]), np.array([]))
