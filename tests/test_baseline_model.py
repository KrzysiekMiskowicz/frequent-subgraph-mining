import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deepchem.feat import CircularFingerprint
from rdkit import Chem

tasks, datasets, transformers = dc.molnet.load_bace_classification()
train_dataset, valid_dataset, test_dataset = datasets

mols_train = [Chem.MolFromSmiles(smiles) for smiles in train_dataset.ids if smiles is not None]

featurizer = CircularFingerprint(size=2048, radius=2)
X_train_transformed = featurizer.featurize(mols_train)

y_train_cleaned = np.nan_to_num(train_dataset.y)

clf = RandomForestClassifier()
clf.fit(X_train_transformed, y_train_cleaned[:, 0])

mols_valid = [Chem.MolFromSmiles(smiles) for smiles in valid_dataset.ids if smiles is not None]
X_valid_transformed = featurizer.featurize(mols_valid)
y_pred = clf.predict(X_valid_transformed)

print(f"Accuracy: {accuracy_score(valid_dataset.y[:, 0], y_pred):.2f}")  # 0.68
