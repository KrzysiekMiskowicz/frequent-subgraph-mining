from .moss_feature_extractor import MoSSFeatureExtractor


class MoSSFeatureExtractorLimitedEdges(MoSSFeatureExtractor):

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
            if edges > self.max_edges:
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

