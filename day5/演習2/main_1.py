import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx
from great_expectations.dataset import PandasDataset


class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            local_path = "data/Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            else:
                raise FileNotFoundError(f"{local_path} が存在しません。")

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        data = data.copy()
        columns_to_drop = [
            col
            for col in ["PassengerId", "Name", "Ticket", "Cabin"]
            if col in data.columns
        ]
        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""

        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        required_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"警告: 以下のカラムがありません: {missing_columns}")
            return False, [{"success": False, "missing_columns": missing_columns}]

        try:
            gxp_data = PandasDataset(data)

            results = []
            results.append(
                gxp_data.expect_column_distinct_values_to_be_in_set("Pclass", [1, 2, 3])
            )
            results.append(
                gxp_data.expect_column_distinct_values_to_be_in_set(
                    "Sex", ["male", "female"]
                )
            )
            results.append(
                gxp_data.expect_column_values_to_be_between(
                    "Age", min_value=0, max_value=100
                )
            )
            results.append(
                gxp_data.expect_column_values_to_be_between(
                    "Fare", min_value=0, max_value=600
                )
            )
            results.append(
                gxp_data.expect_column_distinct_values_to_be_in_set(
                    "Embarked", ["C", "Q", "S", ""]
                )
            )

            is_successful = all([r["success"] for r in results])
            return is_successful, results
        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        preprocessor = ModelTester.create_preprocessing_pipeline()

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        return current_metrics["accuracy"] >= baseline_threshold


if __name__ == "__main__":
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データバリデーション
    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")
    if not success:
        for result in results:
            if not result.get("success", True):
                expectation_type = result.get("expectation_config", {}).get(
                    "expectation_type", "不明"
                )
                print(f"異常タイプ: {expectation_type}, 結果: {result}")
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    # モデルのトレーニングと評価
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_params = {"n_estimators": 100, "random_state": 42}
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    model_path = ModelTester.save_model(model)

    baseline_ok = ModelTester.compare_with_baseline(metrics)
    print(f"ベースライン比較: {'合格' if baseline_ok else '不合格'}")
