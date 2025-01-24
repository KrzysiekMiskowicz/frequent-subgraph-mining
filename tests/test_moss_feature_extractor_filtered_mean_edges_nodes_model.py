import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.frequent_subgraph_mining import MoSSFeatureExtractorFilteredMeanEdgesNodes



tasks, datasets, transformers = dc.molnet.load_bace_classification()
train_dataset, valid_dataset, test_dataset = datasets

X_train = train_dataset.X
y_train = train_dataset.y
w_train = train_dataset.w

y_train_cleaned = np.nan_to_num(y_train)
smiles_data = train_dataset.ids

moss_extractor = MoSSFeatureExtractorFilteredMeanEdgesNodes(min_support=0.1)
moss_extractor.fit(smiles_data)
X_train_transformed = moss_extractor.transform(smiles_data)

clf = RandomForestClassifier()
clf.fit(X_train_transformed, y_train_cleaned[:, 0])

X_valid_transformed = moss_extractor.transform(valid_dataset.ids)
y_pred = clf.predict(X_valid_transformed)
print(f"Accuracy: {accuracy_score(valid_dataset.y[:, 0], y_pred):.2f}")  # 0.61
