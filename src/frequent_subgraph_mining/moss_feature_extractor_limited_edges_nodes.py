from .moss_feature_extractor import MoSSFeatureExtractor

from frequent_subgraph_mining import config


class MoSSFeatureExtractorLimitedEdgesNodes(MoSSFeatureExtractor):
    def __init__(self,
                 min_support=0.1,
                 max_edges=10,
                 max_nodes=10,
                 moss_exec_path: str = config.MOSS_EXEC_PATH,
                 input_dir_path: str = config.MOSS_INPUT_DIR_PATH,
                 output_dir_path: str = config.MOSS_OUTPUT_DIR_PATH):
        self.moss_exec_path = moss_exec_path
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path

        self.min_support = min_support
        self.max_edges = max_edges
        self.max_nodes = max_nodes

        self.bag_of_sub_structures = []


    def _parse_sub_file(self, sub_file_path):
        with open(sub_file_path, "r") as f:
            lines = f.readlines()

        if len(lines) == 0:
            raise ValueError("No data in .sub file")
        elif not lines[0].strip() == "id,description,nodes,edges,s_abs,s_rel,c_abs,c_rel":
            raise ValueError("Invalid header of .sub file")

        subgraphs = []
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) != 8:
                raise ValueError(f"Invalid fields number in line: {line}")

            edges = int(parts[3])
            nodes = int(parts[2])
            if edges > self.max_edges or nodes > self.max_nodes:
                continue

            subgraph = {
                "id": int(parts[0]),
                "description": parts[1],
                "nodes": int(parts[2]),
                "edges": edges,
                "s_abs": int(parts[4]),
                "s_rel": float(parts[5]),
                "c_abs": int(parts[6]),
                "c_rel": float(parts[7]),
            }
            subgraphs.append(subgraph)

        return subgraphs

