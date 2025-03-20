import pandas as pd
import numpy as np
import inspect
import sys
import os
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from .const import Const


CONST = Const()
FEATURES_LABELS = CONST.features_labels


def _path_solve(path: str = None) -> str:
    # 現在のスタックフレームを取得
    current_frame = inspect.currentframe()
    # 呼び出し元のスタックフレームに移動
    caller_frame = current_frame.f_back
    # 呼び出し元のグローバル名前空間を取得
    caller_globals = caller_frame.f_globals

    # Jupyter Notebookのパスを取得
    if '__file__' in caller_globals:
        # ファイルのパスを表示 (caller_file_path, path: 相対パス)
        if path is not None:
            original_path = Path(caller_globals['__file__'])
            caller_file_path = original_path.parent / path
        else:
            raise ValueError("パスが指定されていません。")
        return caller_file_path
    else:
        if path is None:
            raise ValueError("Jupyter Notebook上でのパスが指定されていません。")
        path = os.path.abspath(path)
        caller_file_path = path
        return caller_file_path


def _read_csv(path) -> pd.DataFrame:
    # ファイルのパスを取得
    caller_file_path = _path_solve(path)
    # print(caller_file_path)

    # ファイルの読み込み (欠損値の削除)
    df = pd.read_csv(caller_file_path).replace([np.inf, -np.inf], np.nan).dropna(how="any").dropna(how="all", axis=1)
    df = df.drop_duplicates()

    return df


def _min_max_normalization(p):
    # 正規化
    min_p = p.min()
    max_p = p.max()
    return (p - min_p) / (max_p - min_p)


def _balance_data(smotenc_labels: list[str], train: pd.DataFrame):
    y_train = train["Number Label"]
    X_train = train.drop(columns=["Number Label"])

    smote_nc = SMOTENC(
        categorical_features=[X_train.columns.get_loc(label) for label in smotenc_labels],
        random_state=42,
        k_neighbors=3
    )
    X_train, y_train = smote_nc.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_train)
    y_resampled = pd.DataFrame(y_train, columns=["Number Label"])
    train_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    print("データのバランス調整が完了しました。")

    return train_resampled


def data_preprocessing(train_data, test_data = None, categorical_index: list[str] = None, binary_normal_label: str = None):
    FEATURES_LABELS = CONST.features_labels

    # pattern: (path, path) or (path, None)
    if test_data is not None:
        # ファイルの読み込み
        df = _read_csv(train_data)
        train_len = len(df)
        df_test = _read_csv(test_data)
        # データの結合
        df = pd.concat([df, df_test], axis=0)
    else:
        # ファイルの読み込み
        df = _read_csv(train_data)
    
    # データの前処理
    df = df.filter(items=FEATURES_LABELS + ["Label"])
    label_list = df["Label"].unique()
    if binary_normal_label is not None:
        if binary_normal_label not in label_list:
            raise ValueError("正常データのラベルがデータに存在しません。")
        df["Number Label"] = df["Label"].apply(lambda x: 0 if x == binary_normal_label else 1)
    else:
        df["Number Label"] = df["Label"].apply(lambda x: np.where(label_list == x)[0][0])
    df = df.drop(columns=["Label"])

    print("データの前処理を開始します。")

    # One-Hot Encoding
    categorical_list = [label for label in categorical_index]

    if categorical_index is not None:
        ohe = OneHotEncoder(sparse_output=False)
        df_ohe = ohe.fit_transform(df[categorical_list])
        df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out(categorical_list))

        # インデックス重複対策
        df = df.drop(columns=categorical_list).reset_index(drop=True)
        df_ohe = df_ohe.reset_index(drop=True)

        # カラム名重複対策
        df_ohe = df_ohe.loc[:, ~df_ohe.columns.duplicated()]

        # 安全な結合
        df = pd.concat([df, df_ohe], axis=1)
        ohe_labels = ohe.get_feature_names_out(categorical_list).tolist()
    
    print("One-Hot Encodingが完了しました。")

    # normalization_label = FEATURES_LABELS - categorical_index
    normalization_label = [label for label in FEATURES_LABELS if label not in categorical_list]
    # 正規化
    for label in normalization_label:
        df.loc[:, label] = _min_max_normalization(df[label]).astype(df[label].dtype)
    
    print("正規化が完了しました。")

    if test_data is not None:
        train = df.iloc[:train_len - 1]
        test = df.iloc[train_len - 1:]

        train = train.dropna(how="any")
        test = test.dropna(how="any")

        if categorical_index is not None:
            train_resampled = _balance_data(ohe_labels, train)
            return train_resampled, test, label_list
        else:
            return train, test, label_list
    else:
        df = df.dropna(how="any")
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        if categorical_index is not None:
            train_resampled = _balance_data(ohe_labels, train)
            return train_resampled, test, label_list
        else:
            return train, test, label_list