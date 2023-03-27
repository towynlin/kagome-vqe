from dataclasses import dataclass, field
import numpy as np
from time import time, strftime
from typing import Any, List, Tuple


def relative_error(val: float) -> float:
    return abs((-18.0 - val) / -18.0)


@dataclass
class VQELog:
    first_time: float = 0.0
    last_time: float = 0.0
    values: list[float] = field(default_factory=list)
    parameters: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.first_time = time()

    def update(
        self, count: int, parameters: np.ndarray, value: float, meta: dict[str, Any]
    ):
        self.last_time = time()
        self.values.append(value)
        self.parameters.append(parameters)
        t = strftime("%m/%d %H:%M:%S%z")
        job_num = ((count - 1) // 4) + 1
        roto_subscript = ["0?", "+?", "-?", "best"][(count - 1) % 4]
        aspcr = self.avg_seconds_per_circuit_run()
        eh = self.eight_error_plus_hours(value)
        print(
            f"{t} Ran circuit {job_num}[{roto_subscript}]\testimated value: {value: 08.04f}\tavg sec/run: {aspcr:05.02f}\t8E+H: {eh:07.04f}",
            flush=True,
        )
        if count % 48 == 0:
            print(f"\nParameters: {parameters}\n")

    def rotoselect_update(
        self,
        iteration: int,
        d: int,
        gate_change: Tuple[bool, str, str],
        gate_names: List[str],
        parameters: np.ndarray,
        energy: float,
    ):
        self.last_time = time()
        self.values.append(energy)
        self.parameters.append(parameters)
        t = strftime("%m/%d %H:%M:%S%z")
        asprcb = self.avg_seconds_per_rotoselect_callback()
        print(
            f"{t} Iteration {iteration} gate {d}: {gate_change}\tenergy: {energy: 012.08f}\tavg sec/cb: {asprcb:05.02f}",
            flush=True,
        )
        if len(self.values) % 24 == 0:
            print(f"\nParameters: {parameters}\n")
            print(f"Gates: {gate_names}\n")

    def avg_seconds_per_circuit_run(self) -> float:
        num_intervals = len(self.values) - 1
        if num_intervals == 0:
            return 0
        else:
            diff = self.last_time - self.first_time
            return diff / num_intervals

    def avg_seconds_per_rotoselect_callback(self) -> float:
        # Behavior is different, thus the different name,
        # but the calculation is identical.
        return self.avg_seconds_per_circuit_run()

    def eight_error_plus_hours(self, value: float) -> float:
        err = relative_error(value)
        hours = (time() - self.first_time) / 3600
        return 8 * err + hours
