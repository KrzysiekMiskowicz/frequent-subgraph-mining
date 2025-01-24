from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from frequent_subgraph_mining import MoSSFeatureExtractor


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


smiles_data = [
    'a,0,CCS(O)(O)N',
    'b,0,CCS(O)(C)N',
    'c,0,CS(O)(C)N',
    'd,0,CCS(=N)N',
    'e,0,CS(=N)N',
    'f,0,CS(=N)O',
]
labels = [1, 0, 1, 0, 0, 1]

pipeline = Pipeline([
    ("moss", MoSSFeatureExtractor(min_support=0.5, max_edges=5)),
    ("classifier", RandomForestClassifier())
])

pipeline.fit(smiles_data, labels)
predictions = pipeline.predict(smiles_data)

print(f"Predictions: {predictions:.2f}")
