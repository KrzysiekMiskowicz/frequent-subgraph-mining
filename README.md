## Frequent Subgraph Mining for Molecular Classification
This project focuses on applying Frequent Subgraph Mining (FSM) to molecular graph classification. FSM identifies frequent substructures in data, and its discriminative variant extracts subgraphs that are frequent in one class (e.g., positive) but rare in another (e.g., negative). This approach is particularly useful for molecular graphs, where frequent subgraphs often correspond to important chemical features such as functional groups.

The project integrates MoSS (Molecular Substructure Miner), a Java-based tool, with Python for machine learning workflows. The extracted substructures are used as features for graph classification, providing a more interpretable and flexible alternative to traditional molecular fingerprints like Circular Fingerprints.

### Key Features:
1. MoSSFeatureExtractor Base Class:
    * Provides a scikit-learn-compatible implementation of fit and transform.
    * Uses MoSS to extract frequent subgraphs as features.
    * Supports configurable hyperparameters such as min_support and max_edges.
    * Handles file-based input/output with MoSS and parses its output.

2. MoSSFeatureExtractorFilteredMeanEdgesNodes:
   * An advanced feature extractor that filters subgraphs based on the median number of edges and nodes, ensuring a focus on relevant substructures.
   * Allows additional control via edge_tolerance and node_tolerance parameters.

3. Baseline Model Comparison:
   * Compares the FSM-based feature extraction against baseline models using Circular Fingerprints from RDKit.

4. Benchmark Testing:
   * Evaluates models on multiple datasets from DeepChem MoleculeNet, such as:
        * bace_classification
        * tox21
        * toxcast
        * sider
        * muv
        * clintox
        * hiv

5. Classification:
   * Random Forest is used to classify molecular graphs based on extracted features

6. Results:

| Dataset            | MoSS Model Accuracy | Baseline Model Accuracy  |
|--------------------|--------------------|--------------------------|
| bace_classification | 0.61       | 0.66                     |
| tox21              | 0.96       | 0.97                     |
| toxcast            | 0.94       | 0.96                     |
| sider              | 0.59       | 0.63                     |
| muv                | 1.00       | 1.00                     |
| hiv                | 0.97       | 0.98                     |
