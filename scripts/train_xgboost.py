import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from time import time
import joblib

def preprocess_data(df):
    """Preprocess the Titanic dataset for model training."""
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    
    return X, y

def train_model(dtrain, params, num_round):
    """Train the XGBoost model with the specified parameters."""
    start_time = time()
    model = xgb.train(params, dtrain, num_boost_round=num_round)
    print(f"Training time: {time() - start_time:.2f} seconds")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="Path to training data CSV")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--num_round", type=int, default=100)

    args = parser.parse_args()

    # Detect if running on SageMaker or locally
    input_data_path = os.path.join(os.environ.get("SM_CHANNEL_TRAIN", "."), args.train)
    output_data_path = os.path.join(os.environ.get("SM_MODEL_DIR", "."), "xgboost-model")

    # Load and preprocess data
    df = pd.read_csv(input_data_path)
    X, y = preprocess_data(df)

    # Split data for evaluation purposes
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Train size: {X_train.shape}')
    print(f'Val size: {X_val.shape}')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up XGBoost parameters for training
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": "logloss",
    }

    # Train the model
    model = train_model(dtrain, params, args.num_round)
    joblib.dump(model, output_data_path)
    print(f"Model saved to {output_data_path}")
