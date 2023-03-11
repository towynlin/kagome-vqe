from kagomevqe import HeisenbergModel
from qiskit.opflow import PauliSumOp
from qiskit_nature.second_q.mappers import LogarithmicMapper
from qiskit_nature.second_q.hamiltonians.lattices import Lattice
import rustworkx as rx


class KagomeHamiltonian:
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
    def pauli_sum_op() -> PauliSumOp:
        graph_16 = KagomeHamiltonian.graph()
        kagome_unit_cell_16 = Lattice(graph_16)
        heis_16 = HeisenbergModel(kagome_unit_cell_16)
        log_mapper = LogarithmicMapper()
        return 4 * log_mapper.map(heis_16.second_q_op().simplify())  # type: ignore
