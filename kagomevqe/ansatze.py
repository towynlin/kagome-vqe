from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.providers.fake_provider import FakeGuadalupeV2


class GuadalupeEfficientSU2(QuantumCircuit):
    def __init__(self):
        ansatz = EfficientSU2(12, reps=4)
        guadalupe = FakeGuadalupeV2()
        q_layout = [1, 2, 3, 5, 8, 11, 14, 13, 12, 10, 7, 4]
        ansatz = transpile(
            ansatz,
            backend=guadalupe,
            initial_layout=q_layout,
            coupling_map=guadalupe.coupling_map,
        )
        self = ansatz


class GuadalupeExpressibleJosephsonSampler(QuantumCircuit):
    """Circuit 11 from Sim, Johnson, and Aspuru-Guzik's 2019 paper
    "Expressibility and entangling capability of parameterized
    quantum circuits for hybrid quantum-classical algorithms"
    https://arxiv.org/abs/1905.10876.

    The authors derived circuit 11 from Michael Geller's 2017 paper
    "Sampling and scrambling on a chain of superconducting qubits"
    https://arxiv.org/abs/1711.11026,
    a study of the performance of Josephson sampler circuits as a practical
    means to embed large amounts of classical data in a quantum state.

    Among the 19 circuits in Sim et al., this ansatz provides the best balance
    of high expressibility, low parameter count (96), and low depth (24).
    The RZ and CX gates are native to Guadalupe.
    Only RY requires transpilation to SX-RZ-SX-RZ.
    """

    def __init__(self):
        super().__init__(16)
        entangler_map = [
            [(1, 2), (3, 5), (8, 11), (14, 13), (12, 10), (7, 4)],
            [(2, 3), (5, 8), (11, 14), (13, 12), (10, 7), (4, 1)],
        ]
        self += TwoLocal(
            16,
            ["ry", "rz"],
            "cx",
            entangler_map,
            reps=4,
            skip_unentangled_qubits=True,
            skip_final_rotation_layer=True,
        ).decompose()

    def transpiled(self) -> QuantumCircuit:
        c11_trans = transpile(self, backend=FakeGuadalupeV2())
        # Narrow the type
        assert isinstance(c11_trans, QuantumCircuit)
        return c11_trans


class GuadalupeKagomeRotationalSymmetry(QuantumCircuit):
    def __init__(self):
        super().__init__(16)
        self.sx([1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14])
        self.sx([2, 4, 5, 10, 11, 13])
        self.rz(Parameter("𝜃1"), 1)
        self.rz(Parameter("𝜃3"), 3)
        self.rz(Parameter("𝜃7"), 7)
        self.rz(Parameter("𝜃8"), 8)
        self.rz(Parameter("𝜃12"), 12)
        self.rz(Parameter("𝜃14"), 14)
        self.cx(1, 2)
        self.cx(3, 5)
        self.cx(8, 11)
        self.cx(14, 13)
        self.cx(12, 10)
        self.cx(7, 4)
