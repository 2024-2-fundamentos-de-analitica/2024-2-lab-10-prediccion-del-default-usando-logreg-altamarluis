import os
import glob
import gzip
import json
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import GridSearchCV
    
def _create_output_directory(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)
    
def _save_model(path, estimator):
    _create_output_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)
    

def pregunta01():
    data_test = pd.read_csv("./files/input/test_data.csv.zip",index_col=False,compression="zip")
    data_train = pd.read_csv("./files/input/train_data.csv.zip",index_col = False,compression ="zip")
    
    def cleanse(df):
        df_copy = df.copy()
        df_copy = df_copy.rename(columns={'default payment next month' : "default"})
        df_copy = df_copy.drop(columns=["ID"])
        df_copy = df_copy.loc[df["MARRIAGE"] != 0]
        df_copy = df_copy.loc[df["EDUCATION"] != 0]
        df_copy["EDUCATION"] = df_copy["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
        df_copy = df_copy.dropna()
        return df_copy
    
    data_train = cleanse(data_train)
    data_test = cleanse(data_test)

    x_train, y_train = data_train.drop(columns=["default"]), data_train["default"]
    x_test, y_test = data_test.drop(columns=["default"]), data_test["default"]
    
    def f_pipeline():
        categorical_features=["SEX","EDUCATION","MARRIAGE"]
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), categorical_features) 
            ],
            remainder=MinMaxScaler() 
        )
        selectkbest = SelectKBest(score_func=f_regression, k = 10)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("feature_selection", selectkbest),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
            ],
            verbose=False,
        )
        return pipeline
    
    pipeline = f_pipeline()

    def create_estimator(pipeline, x_train):
        param_grid = {
        "feature_selection__k": range(1, len(x_train.columns) + 1),
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"]
    }
        return  GridSearchCV(pipeline, 
                            param_grid, 
                            cv=10,
                            scoring="balanced_accuracy",
                            n_jobs=-1,
                            refit=True)
    
    estimator = create_estimator(pipeline, x_train)
    estimator.fit(x_train, y_train)

    _save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    def calc_metrics(dataset_type, y_true, y_pred):
        return {
            "type": "metrics",
            "dataset": dataset_type,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
    
    def matrix_calc(dataset_type, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return {
            "type": "cm_matrix",
            "dataset": dataset_type,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        }

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = calc_metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = calc_metrics("train", y_train, y_train_pred)

    test_confusion_metrics = matrix_calc("test", y_test, y_test_pred)
    train_confusion_metrics = matrix_calc("train", y_train, y_train_pred)

    os.makedirs("files/output/", exist_ok=True)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")


if __name__ == "__main__":
    pregunta01()