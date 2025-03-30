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


def _min_max_normalization(p, debug: bool = False):
    """Normalize values to 0-1 range with better handling of edge cases"""
    min_p = p.min()
    max_p = p.max()
    
    # Check if min == max (would cause division by zero)
    if min_p == max_p:
        if debug:
            print(f"Warning: Column has all identical values ({min_p}), returning zeros")
        return pd.Series(0, index=p.index)
    
    # Check for NaN values
    if p.isna().any():
        if debug:
            print(f"Warning: Column contains NaN values before normalization")
    
    # Perform normalization
    normalized = (p - min_p) / (max_p - min_p)
    
    # Final check for infinite values that might have been created
    if np.isinf(normalized).any():
        if debug:
            print(f"Warning: Normalization produced infinite values, replacing with NaN")
        normalized = normalized.replace([np.inf, -np.inf], np.nan)
    
    return normalized


def _balance_data(smotenc_labels: list[str], train: pd.DataFrame):
    y_train = train["Number Label"]
    X_train = train.drop(columns=["Number Label"])

    smote_nc = SMOTENC(
        categorical_features=[X_train.columns.get_loc(label) for label in smotenc_labels],
        random_state=42,
        k_neighbors=3,
        sampling_strategy="minority",
    )
    X_train, y_train = smote_nc.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(X_train)
    y_resampled = pd.DataFrame(y_train, columns=["Number Label"])
    train_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    print("=", end="")

    return train_resampled


FEATURES_LABELS = CONST.features_labels


def data_preprocessing(train_data, test_data = None, categorical_index: list[str] = None, binary_normal_label: str = None, balance: bool = False, debug: bool = False):
    print("- データの読み込み")
    if test_data is not None:
        # ファイルの読み込み
        df = _read_csv(train_data)
        train_len = len(df)
        df_test = _read_csv(test_data)
        # データの結合
        df = pd.concat([df, df_test], axis=0)
    else:
        # ファイルの読み込み
        print("\t- 学習データ")
        df = _read_csv(train_data)
        train_len = None
    
    # print("<データの読み込みが完了しました。>")

    preprocessing_from_data(df, train_len, categorical_index=categorical_index, binary_normal_label=binary_normal_label, balance=balance, debug=debug)

def filter_numberic(df: pd.DataFrame, debug: bool = False, binary_normal_label: str = None) -> pd.DataFrame:
    df = df.filter(items=FEATURES_LABELS + ["Label"])
    label_list = df["Label"].unique()
    if binary_normal_label is not None:
        if binary_normal_label not in label_list:
            raise ValueError("正常データのラベルがデータに存在しません。")
        df["Number Label"] = df["Label"].apply(lambda x: 0 if x == binary_normal_label else 1)
    else:
        df["Number Label"] = df["Label"].apply(lambda x: np.where(label_list == x)[0][0])
    df = df.drop(columns=["Label"])

    return df, label_list


def ohe_hot_encoding(df: pd.DataFrame, categorical_list: list[str] = None) -> pd.DataFrame:
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
    
    return df, ohe_labels


def normalization_label(df: pd.DataFrame, categorical_list: list[str] = None, debug: bool = False) -> list[str]:
    # normalization_label = FEATURES_LABELS - categorical_index
    normalization_label = [label for label in FEATURES_LABELS if label not in categorical_list]
    # 正規化
    for label in normalization_label:
        # Replace the current line with:
        normalized_values = _min_max_normalization(df[label], debug=debug)
        # Check if original column was integer type
        if np.issubdtype(df[label].dtype, np.integer):
            # For integer columns, we need to handle NaNs before conversion
            if debug:
                print(f"Column {label} has integer type, checking for NaNs before conversion")
            if normalized_values.isna().any():
                # Either fill NaNs or keep as float
                if debug:
                    print(f"Warning: NaNs found in normalized values for {label}, keeping as float")
                df.loc[:, label] = normalized_values
            else:
                # Safe to convert to original type
                df.loc[:, label] = normalized_values.astype(df[label].dtype)
        else:
            # For float columns, no issue with NaNs
            df.loc[:, label] = normalized_values
    
    return df

def preprocessing_until_all(df: pd.DataFrame, categorical_index: list[str] = None, binary_normal_label: str = None, debug: bool = False):
    # データの前処理

    df, label_list = filter_numberic(df, debug=debug, binary_normal_label=binary_normal_label)
    
    print("=", end="")

    # print("- one-hot encoding")
    # One-Hot Encoding
    categorical_list = [label for label in categorical_index]

    if categorical_index is not None:
        df, ohe_labels = ohe_hot_encoding(df, categorical_list=categorical_list)
    
    print("=", end="")

    df = normalization_label(df, categorical_list=categorical_list, debug=debug)
    
    print("= done")
    
    return df, label_list, ohe_labels


def preprocessing_from_data(df: pd.DataFrame, train_len: int = None, categorical_index: list[str] = None, binary_normal_label: str = None, balance: bool = False, debug: bool = False):
    # データの前処理

    df, label_list, ohe_labels = preprocessing_until_all(df, categorical_index=categorical_index, binary_normal_label=binary_normal_label, debug=debug)

    if train_len is not None:
        train = df.iloc[:train_len - 1]
        test = df.iloc[train_len - 1:]

        train = train.dropna(how="any")
        test = test.dropna(how="any")

        if categorical_index is not None and balance:
            # print("- データのバランス調整")
            train_resampled = _balance_data(ohe_labels, train)
            print("= done")
            return train_resampled, test, label_list
        else:
            print("= done")
            return train, test, label_list
    else:
        df = df.dropna(how="any")
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        if categorical_index is not None and balance:
            # print("- データのバランス調整")
            train_resampled = _balance_data(ohe_labels, train)
            print("= done")
            return train_resampled, test, label_list
        else:
            print("= done")
            return train, test, label_list