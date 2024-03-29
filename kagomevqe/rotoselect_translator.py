from qiskit.circuit import Gate
from qiskit.circuit.library import RZGate, RYGate, RXGate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass
from typing import List, Tuple, Type


class RotoselectTranslator(TransformationPass):
    """A transpiler pass to replace an individual rotation gate."""

    def __init__(self):
        super().__init__()
        self.parameter_index = 0
        self.replacement_gate = RYGate
        self._last_did_change = False
        self._last_old_gate = "rz"
        self._last_new_gate = "rz"
        self._last_parameterized_gate_name_list = []

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

    @property
    def last_substitution(self) -> Tuple[bool, str, str]:
        return (self._last_did_change, self._last_old_gate, self._last_new_gate)

    @property
    def parameterized_gate_name_list(self) -> List[str]:
        return self._last_parameterized_gate_name_list

    def run(self, dag: DAGCircuit):
        """Run the pass."""

        parameter_count = 0
        self._last_parameterized_gate_name_list = []
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

                if isinstance(gate.params[0], ParameterExpression):
                    has_free_param = len(gate.params[0].parameters) > 0
                    if has_free_param:
                        if parameter_count == self.parameter_index:
                            if isinstance(gate, self.replacement_gate):
                                self._last_did_change = False
                                self._last_old_gate = gate.name
                                self._last_new_gate = gate.name
                            else:
                                # This is the node to replace.
                                param = gate.params[0]
                                replacement = self.replacement_gate(param)
                                dag.substitute_node(node, replacement)
                                self._last_did_change = True
                                self._last_old_gate = gate.name
                                self._last_new_gate = replacement.name
                            self._last_parameterized_gate_name_list.append(
                                self._last_new_gate
                            )
                        else:
                            self._last_parameterized_gate_name_list.append(gate.name)

                        parameter_count += num_params

        return dag
