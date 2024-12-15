import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation

# データの読み込み
data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')

# カテゴリ変数のエンコーディング
categorical_cols = data_train.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    data_train[col] = data_train[col].astype(str)
    data_train[col] = le.fit_transform(data_train[col])
    data_test[col] = data_test[col].astype(str)
    data_test[col] = le.transform(data_test[col])

# 特徴量リストを作成（SK_ID_CURR と TARGET を除外）
num_cols = [col for col in data_train.columns if data_train[col].dtype in ['float64', 'int64'] and col not in ['TARGET', 'SK_ID_CURR']]

# 欠損値の補完
for col in num_cols:
    if data_train[col].isnull().sum() > 0:
        data_train[col] = data_train[col].fillna(data_train[col].median())
    if data_test[col].isnull().sum() > 0:
        data_test[col] = data_test[col].fillna(data_train[col].median())


# 特徴量と目的変数の分離
X = data_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = data_train['TARGET']

# テストデータは TARGET が存在しない
X_test = data_test.drop(['SK_ID_CURR'], axis=1)


# データ分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# LightGBM モデル構築
d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Early Stoppingとログコールバックを追加
model = lgb.train(
    params,
    d_train,
    valid_sets=[d_train, d_valid],
    num_boost_round=1000,
    callbacks=[
        early_stopping(stopping_rounds=50),  # Early Stoppingの設定
        log_evaluation(period=50)           # 50ラウンドごとにログ出力
    ]
)

# モデル評価
y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, y_pred_valid)
print(f"Validation AUC: {auc:.4f}")

# テストデータに対する予測
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
submission = pd.DataFrame({'SK_ID_CURR': data_test['SK_ID_CURR'], 'TARGET': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("提出ファイルを保存しました: submission.csv")
