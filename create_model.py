import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(iris.data, iris.target)

joblib.dump(model, "models/iris_random_forest_model.joblib")

print("Model saved successfully!")