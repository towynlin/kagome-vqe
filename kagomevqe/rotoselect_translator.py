from qiskit.circuit import Gate
from qiskit.circuit.library import RZGate, RYGate, RXGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass
from typing import Type


class RotoselectTranslator(TransformationPass):
    """A transpiler pass to replace an individual rotation gate."""

    def __init__(self):
        super().__init__()
        self.parameter_index = 0
        self.replacement_gate = RYGate

    @property
    def parameter_index(self) -> int:
        return self._parameter_index

    @parameter_index.setter
    def parameter_index(self, value: int) -> None:
        self._parameter_index = value

    @property
    def replacement_gate(self) -> Type[RZGate | RYGate | RXGate]:
        return self._replacement_gate

    @replacement_gate.setter
    def replacement_gate(self, value: Type[RZGate | RYGate | RXGate]) -> None:
        self._replacement_gate = value

    def run(self, dag: DAGCircuit):
        """Run the pass."""

        parameter_count = 0
        for node in dag.gate_nodes():
            # Type narrowing
            assert isinstance(node, DAGOpNode)
            gate = node.op
            assert isinstance(gate, Gate)

            num_params = len(gate.params)
            if num_params > 0:
                # Keeping it simple for the moment. Future work.
                if num_params > 1:
                    raise NotImplementedError

                if parameter_count == self.parameter_index:
                    # This is the node to replace.
                    param = gate.params[0]
                    replacement = self.replacement_gate(param)
                    dag.substitute_node(node, replacement)

                parameter_count += num_params

        return dag