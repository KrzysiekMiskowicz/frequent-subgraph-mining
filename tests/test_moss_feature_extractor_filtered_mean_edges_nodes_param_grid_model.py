import numpy as np
import deepchem as dc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.frequent_subgraph_mining import MoSSFeatureExtractorFilteredMeanEdgesNodes


tasks, datasets, transformers = dc.molnet.load_bace_classification()
train_dataset, valid_dataset, test_dataset = datasets

X_train = train_dataset.X
y_train = train_dataset.y
w_train = train_dataset.w

y_train_cleaned = np.nan_to_num(y_train)
smiles_data = train_dataset.ids

pipeline = Pipeline([
    ('moss', MoSSFeatureExtractorFilteredMeanEdgesNodes()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'moss__min_support': [0.05, 0.1, 0.2],
    'moss__edge_tolerance': [1, 3],
    'moss__node_tolerance': [1, 3],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1
)

grid_search.fit(smiles_data, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best result: {grid_search.best_score_}")

# Best params: {
# 'clf__max_depth': None,
# 'clf__n_estimators': 50,
# 'moss__edge_tolerance': 1,
# 'moss__min_support': 0.05,
# 'moss__node_tolerance': 1
# }
# Best result: 0.7107358998927188
