import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
MIN_ACCURACY_THRESHOLD = 0.75  # 許容最低精度
MAX_INFERENCE_TIME = 1.0  # 許容最大推論時間(秒)
PERFORMANCE_DEGRADATION_TOLERANCE = 0.05  # 許容精度低下率

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """テスト用データセットを読み込む
    
    Returns:
        pd.DataFrame: Titanicデータセット
    """
    if not os.path.exists(DATA_PATH):
        from sklearn.datasets import fetch_openml

        titanic = fetch_openml("titanic", version=1, as_frame=True)
        df = titanic.data
        df["Survived"] = titanic.target

        # 必要なカラムのみ選択
        df = df[
            ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]
        ]

        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor() -> ColumnTransformer:
    """前処理パイプラインを定義
    
    Returns:
        ColumnTransformer: 前処理済みのデータ
    """
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data: pd.DataFrame, preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """モデルの学習とテストデータの準備
    
    Args:
        sample_data: テスト用データセット
        preprocessor: 前処理パイプライン
    
    Returns:
        Dict: モデルとテストデータを含む辞書
    """
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデル保存前に過去バージョンをバックアップ
    old_model_path = os.path.join(MODEL_DIR, "old_titanic_model.pkl")
    
    if os.path.exists(MODEL_PATH):
        os.replace(MODEL_PATH, old_model_path)  # 既存モデルをold_として移動
    
    # 新モデルを保存
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # 過去モデルが存在する場合は読み込んで返す
    old_model = None
    if os.path.exists(old_model_path):
        with open(old_model_path, "rb") as f:
            old_model = pickle.load(f)

    return {
        "current_model": model,
        "old_model": old_model,  # 過去モデル（存在しない場合はNone）
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train
    }


def test_model_exists():
    """モデルファイルが存在するか確認"""
    assert os.path.exists(MODEL_PATH), f"モデルファイルが存在しません: {MODEL_PATH}"


def test_model_comparison(train_model: Dict[str, Any]):
    """過去バージョンとの性能比較テスト
    
    Args:
        train_model: 学習済みモデルとテストデータ
    """
    result = train_model
    
    if result["old_model"] is None:
        pytest.skip("過去のモデルが存在しないためスキップ")
    
    # 現在のモデルで予測
    current_pred = result["current_model"].predict(result["X_test"])
    current_accuracy = accuracy_score(result["y_test"], current_pred)
    
    # 過去のモデルで予測
    old_pred = result["old_model"].predict(result["X_test"])
    old_accuracy = accuracy_score(result["y_test"], old_pred)
    
    # 精度が許容範囲内であることを確認
    assert current_accuracy >= old_accuracy - PERFORMANCE_DEGRADATION_TOLERANCE, (
        f"モデルの精度が許容範囲以上に低下しました\n"
        f"旧バージョン精度: {old_accuracy:.4f}\n"
        f"新バージョン精度: {current_accuracy:.4f}\n"
        f"許容低下率: {PERFORMANCE_DEGRADATION_TOLERANCE:.2f}"
    )


def test_model_accuracy(train_model: Dict[str, Any]):
    """モデルの精度を検証
    
    Args:
        train_model: 学習済みモデルとテストデータ
    """
    model = train_model["current_model"]
    X_test = train_model["X_test"]
    y_test = train_model["y_test"]

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= MIN_ACCURACY_THRESHOLD, (
        f"モデルの精度が最低閾値を下回っています\n"
        f"現在の精度: {accuracy:.4f}\n"
        f"最低要求精度: {MIN_ACCURACY_THRESHOLD:.2f}"
    )


def test_model_inference_time(train_model: Dict[str, Any]):
    """モデルの推論時間を検証
    
    Args:
        train_model: 学習済みモデルとテストデータ
    """
    model = train_model["current_model"]
    X_test = train_model["X_test"]

    # 推論時間の計測
    start_time = time.perf_counter()  # より高精度な計測
    model.predict(X_test)
    end_time = time.perf_counter()

    inference_time = end_time - start_time

    assert inference_time < MAX_INFERENCE_TIME, (
        f"推論時間が許容時間を超えています\n"
        f"現在の推論時間: {inference_time:.4f}秒\n"
        f"許容最大時間: {MAX_INFERENCE_TIME}秒"
    )


def test_model_reproducibility(sample_data: pd.DataFrame, preprocessor: ColumnTransformer):
    """モデルの再現性を検証
    
    Args:
        sample_data: テスト用データセット
        preprocessor: 前処理パイプライン
    """
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 同じパラメータで2つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    assert np.array_equal(predictions1, predictions2), (
        "同じパラメータと乱数シードで学習したモデルが異なる予測結果を出力しました\n"
        "モデルの再現性が保証されていません"
    )


def test_model_feature_importance(train_model: Dict[str, Any]):
    """特徴量の重要度が適切であることを検証
    
    Args:
        train_model: 学習済みモデルとテストデータ
    """
    model = train_model["current_model"]
    
    # 特徴量重要度を取得
    classifier = model.named_steps['classifier']
    assert hasattr(classifier, 'feature_importances_'), "モデルに特徴量重要度属性がありません"
    
    importances = classifier.feature_importances_
    
    # 重要度が正しい形状であることを確認
    preprocessor = model.named_steps['preprocessor']
    transformed = preprocessor.transform(train_model["X_train"].iloc[:1])
    assert len(importances) == transformed.shape[1], (
        f"特徴量重要度の数が不正です\n"
        f"期待される数: {transformed.shape[1]}\n"
        f"実際の数: {len(importances)}"
    )
    
    # 重要度が適切な範囲にあることを確認
    assert np.all(importances >= 0), "特徴量重要度に負の値が含まれています"
    assert np.any(importances > 0), "全ての特徴量重要度が0です"


def test_data_quality(sample_data: pd.DataFrame):
    """データの品質を検証
    
    Args:
        sample_data: テスト用データセット
    """
    # 欠損値チェック
    missing_values = sample_data.isnull().sum()
    high_missing_cols = missing_values[missing_values > 0.5 * len(sample_data)]
    assert len(high_missing_cols) == 0, (
        f"欠損値が多すぎるカラムがあります:\n{high_missing_cols}"
    )
    
    # ターゲット変数のバランスチェック
    target_dist = sample_data["Survived"].value_counts(normalize=True)
    assert 0.3 < target_dist[0] < 0.7, (
        f"ターゲット変数のクラス不均衡が大きすぎます:\n{target_dist}"
    )