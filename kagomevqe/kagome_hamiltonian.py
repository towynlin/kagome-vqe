from kagomevqe import HeisenbergModel
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature import settings
from qiskit_nature.second_q.mappers import LogarithmicMapper
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
import rustworkx as rx


settings.use_pauli_sum_op = False


class KagomeHamiltonian:
    @staticmethod
    def ground_state_energy() -> float:
        return -18

    @staticmethod
    def graph() -> rx.PyDAG:
        # Edge weight
        t = 1.0

        graph_16 = rx.PyGraph(multigraph=False)  # type: ignore
        graph_16.add_nodes_from(range(16))
        edge_list = [
            (1, 2, t),
            (2, 3, t),
            (3, 5, t),
            (5, 8, t),
            (8, 11, t),
            (11, 14, t),
            (14, 13, t),
            (13, 12, t),
            (12, 10, t),
            (10, 7, t),
            (7, 4, t),
            (4, 1, t),
            (4, 2, t),
            (2, 5, t),
            (5, 11, t),
            (11, 13, t),
            (13, 10, t),
            (10, 4, t),
        ]
        graph_16.add_edges_from(edge_list)
        return graph_16

    @staticmethod
    def pauli_sum_op() -> SparsePauliOp:
        graph_16 = __class__.graph()
        kagome_unit_cell_16 = Lattice(graph_16)
        heis_16 = HeisenbergModel(kagome_unit_cell_16)
        log_mapper = LogarithmicMapper()
        mapped_op = log_mapper.map(heis_16.second_q_op().simplify())
        assert isinstance(mapped_op, SparsePauliOp)
        return mapped_op._multiply(4)


class Kagome16AsymmetricHamiltonian:
    @staticmethod
    def ground_state_energy() -> float:
        return -25.87852551

    @staticmethod
    def graph() -> rx.PyDAG:
        # Edge weight
        t = 1.0

        graph_16 = rx.PyGraph(multigraph=False)  # type: ignore
        graph_16.add_nodes_from(range(16))
        edge_list = [
            (0, 1, t),
            (0, 2, t),
            (1, 2, t),
            (1, 4, t),
            (1, 7, t),
            (2, 3, t),
            (2, 5, t),
            (3, 5, t),
            (4, 7, t),
            (5, 8, t),
            (5, 9, t),
            (6, 7, t),
            (6, 10, t),
            (7, 10, t),
            (8, 9, t),
            (8, 10, t),
            (8, 12, t),
            (9, 11, t),
            (10, 12, t),
            (11, 14, t),
            (12, 13, t),
            (12, 15, t),
            (13, 14, t),
            (13, 15, t),
        ]
        graph_16.add_edges_from(edge_list)
        return graph_16

    @staticmethod
    def lattice() -> Lattice:
        graph_16 = __class__.graph()
        return Lattice(graph_16)

    @staticmethod
    def pauli_sum_op() -> SparsePauliOp:
        kagome_asymmetric_16 = __class__.lattice()
        heis_16 = HeisenbergModel(kagome_asymmetric_16)
        log_mapper = LogarithmicMapper()
        mapped_op = log_mapper.map(heis_16.second_q_op().simplify())
        assert isinstance(mapped_op, SparsePauliOp)
        return mapped_op._multiply(4)
