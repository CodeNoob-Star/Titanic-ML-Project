import pandas as pd
from joblib import load
from pathlib import Path
from config import FEATURES
from data_preprocessing import preprocess

def predict():
        BASE_DIR = Path(__file__).resolve().parent.parent

        #Load test data
        test_df = pd.read_csv(r"C:\Users\zbook\Documents\Project\Titanic\data\test.csv")
        test_df = preprocess(test_df)

        #Load model
        model_path = BASE_DIR / "models" / "logistic_regression.pkl"
        model = load(model_path)

        #Predict
        prediction = model.predict(test_df[FEATURES])

        #Create Submission
        submission = pd.DataFrame({
                "PassengerId": test_df["PassengerId"],
                "Survived": prediction
        })

        #Save submission
        output_dir = BASE_DIR / "output"
        output_dir.mkdir(exist_ok=True)

        submission_path = output_dir / "submission.csv"

        submission.to_csv(
                submission_path,
                index=False
        )
if __name__ == "__main__":
    predict()

