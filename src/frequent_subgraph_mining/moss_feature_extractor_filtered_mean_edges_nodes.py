from statistics import median, mode

from .moss_feature_extractor import MoSSFeatureExtractor
from frequent_subgraph_mining import config


class MoSSFeatureExtractorFilteredMeanEdgesNodes(MoSSFeatureExtractor):
    def __init__(self,
                 min_support: float = 0.1,
                 max_edges: int = 10,
                 edge_tolerance: int = 1,
                 node_tolerance: int = 1,
                 moss_exec_path: str = config.MOSS_EXEC_PATH,
                 input_dir_path: str = config.MOSS_INPUT_DIR_PATH,
                 output_dir_path: str = config.MOSS_OUTPUT_DIR_PATH):

        super().__init__(min_support, max_edges, moss_exec_path, input_dir_path, output_dir_path)
        self.edge_tolerance = edge_tolerance
        self.node_tolerance = node_tolerance

    def _parse_sub_file(self, sub_file_path):
        with open(sub_file_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            raise ValueError("No data in .sub file")
        elif not lines[0].strip() == "id,description,nodes,edges,s_abs,s_rel,c_abs,c_rel":
            raise ValueError("Invalid header of .sub file")

        subgraphs = []
        edge_counts = []
        node_counts = []

        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) != 8:
                raise ValueError(f"Invalid fields number in line: {line}")

            edge_counts.append(int(parts[3]))
            node_counts.append(int(parts[2]))

            subgraph = {
                "id": int(parts[0]),
                "description": parts[1],
                "nodes": int(parts[2]),
                "edges": int(parts[3]),
                "s_abs": int(parts[4]),
                "s_rel": float(parts[5]),
                "c_abs": int(parts[6]),
                "c_rel": float(parts[7]),
            }
            subgraphs.append(subgraph)

        edge_median = median(edge_counts)
        node_median = median(node_counts)

        # edge_mode = mode(edge_counts)
        # node_mode = mode(node_counts)

        filtered_subgraphs = [
            subgraph for subgraph in subgraphs
            if (edge_median - self.edge_tolerance <= subgraph["edges"] <= edge_median + self.edge_tolerance) and
               (node_median - self.node_tolerance <= subgraph["nodes"] <= node_median + self.node_tolerance)
        ]

        return filtered_subgraphs
