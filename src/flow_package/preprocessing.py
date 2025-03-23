import pandas as pd
import numpy as np
import inspect
import sys
import os
from pathlib import Path
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from .const import Const
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.preprocessing import MinMaxScaler, OneHotEncoder as DaskOneHotEncoder


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
    df = dd.read_csv(caller_file_path).replace([np.inf, -np.inf], np.nan)
    # df = df.dropna(how="any").dropna(how="all", axis=1)
    df = df.dropna(how="any")
    na_cols = [ col for col in df.columns if not df[col].isna().all().compute() ]
    df = df.drop_duplicates()
    df = df[na_cols]

    return df


def _balance_data(smotenc_labels: list[str], train: dd.DataFrame):
    while True:
        ddf = dd.from_pandas(train, npartitions=3)
        for i in range(ddf.npartitions):
            partition = ddf.get_partition(i).compute()[smotenc_labels].nunique().values
            if np.any(partition == 1):
                break
        else:
            break

    print("SMOTE-NCの適用を開始します。")
            
    one_partition = ddf.get_partition(0).compute().drop(columns=["Number Label"])

    feature_index = [
        one_partition.columns.get_loc(label) for label in smotenc_labels
    ]

    smote_nc = SMOTENC(
        categorical_features=feature_index,
        random_state=42,
        k_neighbors=3,
    )

    X_train, y_train = train.map_partitions(
        lambda df: smote_nc.fit_resample(df.drop(columns=["Number Label"]), df["Number Label"]),
        meta={
            "X_train": "object",
            "y_train": "object",
        },
    ).compute()

    X_resampled = dd.DataFrame(X_train)
    y_resampled = dd.DataFrame(y_train, columns=["Number Label"])
    train_resampled = dd.concat([X_resampled, y_resampled], axis=1)

    print("データのバランス調整が完了しました。")

    return train_resampled


def data_preprocessing(train_data, test_data = None, categorical_index: list[str] = None, binary_normal_label: str = None, debug: bool = False):

    print("データの前処理を開始します。")
    print("- データの読み込み")

    # pattern: (path, path) or (path, None)
    if test_data is not None:
        # ファイルの読み込み
        print("\t- 学習データ")
        df = _read_csv(train_data)
        train_len = len(df)
        print("\t- テストデータ")
        df_test = _read_csv(test_data)
        # データの結合
        df = dd.concat([df, df_test], axis=0)
    else:
        # ファイルの読み込み
        print("\t- 学習データ")
        df = _read_csv(train_data)
    
    print("<データの読み込みが完了しました。>")


def _preprocess(df, train_len, categorical_index: list[str] = None, binary_normal_label: str = None):
    FEATURES_LABELS = CONST.features_labels
    # データの前処理
    df = df[FEATURES_LABELS + ["Label"]]
    df = df.dropna(how="any")
    df = df.drop_duplicates()

    label_list = df["Label"].unique().compute()

    if binary_normal_label is not None:
        for label in label_list:
            if label == binary_normal_label:
                break
        else:
            raise ValueError("正常データのラベルがデータに存在しません。")
        df["Number Label"] = (train["Label"] == binary_normal_label).astype(int)
    else:
        df["Number Label"] = (label_list.index.get_loc(df["Label"])).astype(int)
    df = df.drop(columns=["Label"])

    print("- one-hot encoding")
    # One-Hot Encoding
    categorical_list = [label for label in categorical_index]

    if categorical_index is not None:
        ohe = DaskOneHotEncoder(
            categorical_features=categorical_list,
            sparse_output=False
        )
        df = df.categorize(columns=categorical_list)
        df_ohe = ohe.fit_transform(df[categorical_list])

        ohe_labels = df_ohe.columns
        df = df.drop(columns=categorical_list)
        df = dd.concat([df, df_ohe], axis=1)
    
    print("<One-Hot Encodingが完了しました>")

    # normalization_label = FEATURES_LABELS - categorical_index
    normalization_label = [label for label in FEATURES_LABELS if label not in categorical_list]
    print("- 正規化")
    df_compute = df.compute()
    # 正規化
    scaler = MinMaxScaler()
    for label in normalization_label:
        df_compute[label] = scaler.fit_transform(df_compute[label].values.reshape(-1, 1))
    
    df_compute = df_compute.dropna(how="any").drop_duplicates()
    print("<正規化が完了しました>")

    if test_data is not None:
        train = df_compute.iloc[:train_len - 1]
        test = df_compute.iloc[train_len - 1:]

        train = train.dropna(how="any")
        test = test.dropna(how="any")

        if categorical_index is not None:
            print("- データのバランス調整")
            train_resampled = _balance_data(ohe_labels, train)
            return train_resampled, test, label_list
        else:
            return train, test, label_list
    else:
        df = df_compute.dropna(how="any")
        train, test = dask_train_test_split(df, test_size=0.2, random_state=42)

        if categorical_index is not None:
            print("- データのバランス調整")
            train_resampled = _balance_data(ohe_labels, train)
            return train_resampled, test, label_list
        else:
            return train, test, label_list