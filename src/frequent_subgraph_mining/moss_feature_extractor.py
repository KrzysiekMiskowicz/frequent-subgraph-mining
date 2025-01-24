import os
import subprocess

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from rdkit import Chem

from frequent_subgraph_mining import config


class MoSSFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_support=0.1,
                 max_edges=10,
                 moss_exec_path: str = config.MOSS_EXEC_PATH,
                 input_dir_path: str = config.MOSS_INPUT_DIR_PATH,
                 output_dir_path: str = config.MOSS_OUTPUT_DIR_PATH):
        self.moss_exec_path = moss_exec_path
        self.input_dir_path = input_dir_path
        self.output_dir_path = output_dir_path

        self.min_support = min_support
        self.max_edges = max_edges

        self.bag_of_sub_structures = []

    def __post_init__(self):
        os.makedirs(self.input_dir_path, exist_ok=True)
        os.makedirs(self.output_dir_path, exist_ok=True)

    def _write_smiles_to_file(self, smiles_list, input_file_name="input.smiles"):
        input_file_path = os.path.join(self.input_dir_path, input_file_name)
        os.makedirs(self.input_dir_path, exist_ok=True)
        with open(input_file_path, "w") as f:
            for idx, smiles in enumerate(smiles_list):
                f.write(f"{idx},0,{smiles}\n")
        return input_file_path

    def _run_moss(self, input_file_path, output_file_name="output.sub"):
        output_file_path = os.path.join(self.output_dir_path, output_file_name)
        cmd = [
            "java", "-cp", self.moss_exec_path, 'moss.Miner',
            input_file_path,
            output_file_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Moss run finished with an error:\n{e}")
        return output_file_path

    @staticmethod
    def extract_smiles_file_data(smiles_line):
        parts = smiles_line.split(',')
        if len(parts) < 3:
            raise ValueError(f"Invalid SMILES line format: {smiles_line}")

        return parts[2].strip()

    def _smiles_to_graph(self, smiles):
        graph = Chem.MolFromSmiles(smiles)
        if graph is None:
            smiles_data = self.extract_smiles_file_data(smiles)
            graph = Chem.MolFromSmiles(smiles_data)
            if graph is None:
                raise ValueError(f"Invalid SMILES format:\n{smiles}")
        return graph

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

        return subgraphs

    def fit(self, X, y=None):
        input_file = self._write_smiles_to_file(X)
        output_file = self._run_moss(input_file)
        self.bag_of_sub_structures = self._parse_sub_file(output_file)
        return self

    def transform(self, X):
        feature_matrix = []
        for smiles in X:
            mol_graph = self._smiles_to_graph(smiles)
            # mol_graph = Chem.MolFromSmiles(smiles)
            features = []
            for sub_structure in self.bag_of_sub_structures:
                # mol_subgraph = Chem.MolFromSmiles(sub_structure['description'])
                try:
                    mol_subgraph = self._smiles_to_graph(sub_structure['description'])
                    if mol_graph.HasSubstructMatch(mol_subgraph):
                        features.append(1)
                    else:
                        features.append(0)
                except ValueError:
                    features.append(None)
            feature_matrix.append(features)
        return np.array(feature_matrix)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def predict(self, dataset, transformers=[]):

        features = self.transform(dataset.X)
        return features
