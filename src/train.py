import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from pathlib import Path
from config import FEATURES, TARGET
from data_preprocessing import preprocess
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

def train():
    df = pd.read_csv(r"C:\Users\zbook\Documents\Project\Titanic\data\train.csv")

    df = preprocess(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    BASE_DIR = Path(__file__).resolve().parent.parent

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    dump(
        model,
        model_dir / "logistic_regression.pkl"
    )
    coef = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_[0]
    })

    coef = coef.sort_values(
        by="Coefficient",
        ascending=False
    )

    print(coef)
if __name__ == "__main__":
    train()
