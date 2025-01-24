import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deepchem.feat import CircularFingerprint
from rdkit import Chem

from src.frequent_subgraph_mining import MoSSFeatureExtractorFilteredMeanEdgesNodes


def test_my_model(smiles_train, y_train, valid_dataset):
    moss_extractor = MoSSFeatureExtractorFilteredMeanEdgesNodes()
    moss_extractor.fit(smiles_train)
    smiles_train_transformed = moss_extractor.transform(smiles_train)

    clf_moss = RandomForestClassifier()
    clf_moss.fit(smiles_train_transformed, y_train[:, 0])

    smiles_valid_transformed = moss_extractor.transform(valid_dataset.ids)
    y_pred_moss = clf_moss.predict(smiles_valid_transformed)

    return accuracy_score(valid_dataset.y[:, 0], y_pred_moss)


def test_baseline_model(smiles_train, y_train, valid_dataset):
    # === Model bazowy ===
    mols_train = [Chem.MolFromSmiles(smiles) for smiles in smiles_train if smiles is not None]
    featurizer = CircularFingerprint(size=2048, radius=2)
    X_train_baseline = featurizer.featurize(mols_train)

    clf_baseline = RandomForestClassifier()
    clf_baseline.fit(X_train_baseline, y_train[:, 0])

    mols_valid = [Chem.MolFromSmiles(smiles) for smiles in valid_dataset.ids if smiles is not None]
    X_valid_baseline = featurizer.featurize(mols_valid)
    y_pred_baseline = clf_baseline.predict(X_valid_baseline)

    return accuracy_score(valid_dataset.y[:, 0], y_pred_baseline)


benchmark_datasets = [
    "bace_classification",
    "tox21",
    "toxcast",
    "sider",
    "muv",
    "clintox",
    "hiv"
]
results = {}

for dataset_name in benchmark_datasets:
    print(f"\nTesting on dataset: {dataset_name}")
    try:
        tasks, datasets, transformers = getattr(dc.molnet, f"load_{dataset_name}")()
        train_dataset, valid_dataset, test_dataset = datasets

        X_train = train_dataset.X
        y_train = train_dataset.y
        smiles_data = train_dataset.ids
        y_train_cleaned = np.nan_to_num(y_train)

        accuracy_moss = test_my_model(smiles_train=smiles_data,
                                      y_train=y_train_cleaned,
                                      valid_dataset=valid_dataset)
        accuracy_baseline = test_baseline_model(smiles_train=smiles_data,
                                                y_train=y_train_cleaned,
                                                valid_dataset=valid_dataset)

        results[dataset_name] = {
            "Moss Model Accuracy": accuracy_moss,
            "Baseline Model Accuracy": accuracy_baseline
        }

        print(f"Moss Model Accuracy: {accuracy_moss:.2f}")
        print(f"Baseline Model Accuracy: {accuracy_baseline:.2f}")

    except Exception as e:
        print(f"Error testing dataset {dataset_name}: {e}")
        results[dataset_name] = {"Moss Model Accuracy": None, "Baseline Model Accuracy": None}


print("\nFinal Results:")
for dataset, metrics in results.items():
    moss_acc = metrics["Moss Model Accuracy"]
    baseline_acc = metrics["Baseline Model Accuracy"]
    print(f"{dataset}: Moss Model = {moss_acc:.2f}, Baseline Model = {baseline_acc:.2f}")

# {'bace_classification': {'Moss Model Accuracy': 0.609271523178808, 'Baseline Model Accuracy': 0.6556291390728477}, 'tox21': {'Moss Model Accuracy': 0.9565772669220945, 'Baseline Model Accuracy': 0.9719029374201787}, 'toxcast': {'Moss Model Accuracy': 0.9440559440559441, 'Baseline Model Accuracy': 0.9557109557109557}, 'sider': {'Moss Model Accuracy': 0.5874125874125874, 'Baseline Model Accuracy': 0.6293706293706294}, 'muv': {'Moss Model Accuracy': 0.99946288537974, 'Baseline Model Accuracy': 0.99946288537974}, 'clintox': {'Moss Model Accuracy': None, 'Baseline Model Accuracy': None}, 'hiv': {'Moss Model Accuracy': 0.9725261366399222, 'Baseline Model Accuracy': 0.9824945295404814}}
