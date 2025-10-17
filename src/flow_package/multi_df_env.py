import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional
import warnings
import random
from dataclasses import dataclass
warnings.filterwarnings('ignore')


@dataclass
class EnvConfig:
    """
    環境設定用データクラス
    Args:
            data: 処理対象のDataFrame（Noneの場合はサンプルデータを生成）
            window_size: 観測ウィンドウのサイズ
            max_steps: エピソードの最大ステップ数
            render_mode: 描画モード ('human' または 'rgb_array')
    """
    data: pd.DataFrame
    label_column: str
    window_size: int
    max_steps: int
    render_mode: Optional[str]
    # 正規化方法: 'zscore' (従来の全体平均/標準偏差での正規化) または 'rolling' (移動ウィンドウごとの z-score)
    normalize_method: str = 'zscore'
    # 移動ウィンドウのサイズ（normalize_method='rolling' のときに使用）
    rolling_window: int = 50
    test_mode: bool = False  # テストモードフラグ


# TODO: NEED TO CHANGE
class MultiDfEnv(gym.Env):
    """
    カスタムGymnasium環境：DataFrameを使った順次データ処理環境
    
    この環境は時系列データを順番に処理し、エージェントが各ステップで
    行動を選択してデータを処理する学習環境を提供します。
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config: EnvConfig):
        """
        環境の初期化
        
        Args:
            config: 環境設定用データクラス
        """
        super().__init__()
        
        # データの準備
        if config.data is None:
            # self.data = self._generate_sample_data()
            raise ValueError("Data must be provided in config.data")
        else:
            self.data = config.data.copy()

        self.label_column = config.label_column

        self.window_size = config.window_size
        self.max_steps = config.max_steps
        self.render_mode = config.render_mode

        self.data_length = len(self.data)
        self.max_times = self.data_length // self.window_size
        self.end = self.data_length - 1
        self.test_mode = config.test_mode

        # データの特徴量数
        self.n_features = len(self.data.columns) - 1  # ラベル列を除く
        
        label_unique_len = len(self.data[self.label_column].unique())
        self.action_space = spaces.Discrete(label_unique_len)
        
        # 観測空間の定義（正規化された特徴量のウィンドウ + 追加情報）
        obs_shape = (self.window_size * self.n_features + 2,)  # +2 for additional info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # 環境の状態変数
        self.current_step = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.history = []
        
        # データの正規化
        self._normalize_data()
        
    def _generate_sample_data(self) -> pd.DataFrame:
        """サンプルの時系列データを生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        
        # トレンド + ノイズ + 季節性を持つ時系列データ
        trend = np.linspace(100, 150, 500)
        noise = np.random.normal(0, 5, 500)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(500) / 50)
        
        price = trend + noise + seasonal
        
        data = pd.DataFrame({
            'date': dates,
            'price': price,
            'volume': np.random.lognormal(10, 1, 500),
            'ma_5': pd.Series(price).rolling(5).mean(),
            'ma_20': pd.Series(price).rolling(20).mean(),
            'volatility': pd.Series(price).rolling(10).std()
        })
        
        # 欠損値を前方補完
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    
    def _normalize_data(self):
        """データを正規化"""
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        if self.label_column in numeric_columns:
            numeric_columns = numeric_columns.drop(self.label_column)
        self.data_normalized = self.data.copy()
        # 正規化方法に応じて処理
        if getattr(self, 'normalize_method', 'zscore') == 'rolling':
            # 移動ウィンドウで z-score 正規化を行う
            w = getattr(self, 'rolling_window', 50)
            # 各列ごとに rolling mean/std を計算し、(x - mean)/std を適用
            for col in numeric_columns:
                series = self.data[col]
                roll_mean = series.rolling(window=w, min_periods=1, center=False).mean()
                roll_std = series.rolling(window=w, min_periods=1, center=False).std()
                # std が 0 になる可能性があるので安定化
                roll_std = roll_std.replace(0, np.nan)
                normalized = (series - roll_mean) / roll_std
                # 無限大や NaN を適切に扱う（NaN のままにするか 0 に埋めるかは用途により選択できる）
                normalized = normalized.replace([np.inf, -np.inf], np.nan)
                # 端の NaN を前方/後方詰めで埋める（ここでは forward-fill し、残れば backward-fill）
                normalized = normalized.fillna(method='ffill').fillna(method='bfill')
                # 安全のため float にキャスト
                self.data_normalized[col] = normalized.astype(np.float64)
        else:
            # 全体の平均/標準偏差での Z-score 正規化（従来の挙動）
            for col in numeric_columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.data_normalized[col] = (self.data[col] - mean) / (std + 1e-8)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """環境のリセット"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.history = []
        
        observation = self._get_observation()

        if self.test_mode:
            self.end = self.data_length - 1
        else:
            buf = random.randint(1, self.max_times)
            self.end = random.randint(self.window_size, self.window_size * buf)

        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """環境でのステップ実行"""
        # 現在のラベルを取得
        current_label = self.data.iloc[self.current_step][self.label_column]
        
        # 行動に基づく報酬計算
        reward, cm_index = self._calculate_reward(action, current_label)

        # 履歴の記録
        self.history.append({
            'step': self.current_step,
            'confusion_matrix_index': cm_index,
            'reward': reward,
            'total_reward': self.total_reward
        })
        
        # 次のステップへ
        self.current_step += 1
        
        # 終了条件の確認
        terminated = self.current_step >= self.end - 1
        truncated = self.current_step - self.window_size >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info(cm_index=cm_index)
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """現在の観測値を取得"""
        if self.current_step < self.window_size:
            # 初期化時の処理
            window_data = self.data_normalized.iloc[:self.window_size].drop(columns=[self.label_column])
        else:
            # ウィンドウサイズ分のデータを取得
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            window_data = self.data_normalized.iloc[start_idx:end_idx].drop(columns=[self.label_column])
        
        # 数値データのみを抽出してフラット化
        numeric_data = window_data.select_dtypes(include=[np.number]).values.flatten()
        
        # 追加情報（ポジション、ステップ数、総報酬、価格変化率）
        additional_info = np.array([
            self.current_step / self.end,  # 正規化されたステップ数
            self.total_reward / 100.0,  # 正規化された総報酬
        ])
        
        observation = np.concatenate([numeric_data, additional_info]).astype(np.float32)
        return observation
    
    def _calculate_reward(self, action: int, current_label: float) -> float:
        """報酬の計算"""
        reward = 0.0

        """
        current | action | is_correct | reward
        -------|------------ | ------- | -------
        normal | normal | True | +0.5
        normal | attack | False | -2.0
        attack | normal | False | -1.0
        attack | other attack | False | -0.5
        attack | same attack | True | +2.0

        normal : 0, attack : 1 ~
        """
        
        if current_label == 0:  # normal
            if action == 0:  # normal
                reward = 0.5
            else:  # attack
                reward = -2.0
        else:  # attack
            if action == 0:  # normal
                reward = -1.0
            elif action == current_label:  # same attack
                reward = 2.0
            else:  # other attack
                reward = -0.5
        
        self.total_reward += reward
        return reward, (action, current_label)
    
    def _get_info(self, cm_index: Tuple[int, int] = ()) -> Dict:
        """追加情報の取得"""
        return {
            'step': self.current_step,
            'total_reward': self.total_reward,
            'data_progress': self.current_step / self.end if self.end > 0 else 0,
            'confusion_matrix_index': cm_index,
            'sample_data_length': self.end,
        }
    
    def render(self):
        """環境の描画"""
        if self.render_mode == 'human':
            self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self):
        """人間向けの描画"""
        plt.figure(figsize=(12, 8))
        
        # 価格チャート
        plt.subplot(2, 1, 1)
        if self.current_step > 1:
            price_data = self.data['price'][:self.current_step]
            plt.plot(price_data.index, price_data.values, 'b-', alpha=0.7, label='Price')
            
            # ポジション履歴の描画
            for record in self.history:
                if record['action'] == 1:  # Buy
                    plt.scatter(record['step'], record['price'], color='green', s=100, marker='^', label='Buy' if record['step'] == self.history[0]['step'] else '')
                elif record['action'] == 2:  # Sell
                    plt.scatter(record['step'], record['price'], color='red', s=100, marker='v', label='Sell' if record['step'] == self.history[0]['step'] else '')
        
        plt.title('Price Chart with Trading Actions')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 報酬チャート
        plt.subplot(2, 1, 2)
        if self.history:
            rewards = [record['total_reward'] for record in self.history]
            steps = [record['step'] for record in self.history]
            plt.plot(steps, rewards, 'g-', label='Cumulative Reward')
        
        plt.title('Cumulative Reward')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _render_rgb_array(self):
        """RGB配列として描画結果を返す"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 価格チャート
        if self.current_step > 1:
            price_data = self.data['price'][:self.current_step]
            ax1.plot(price_data.index, price_data.values, 'b-', alpha=0.7)
        ax1.set_title('Price Chart')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # 報酬チャート
        if self.history:
            rewards = [record['total_reward'] for record in self.history]
            steps = [record['step'] for record in self.history]
            ax2.plot(steps, rewards, 'g-')
        ax2.set_title('Cumulative Reward')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # RGB配列に変換
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return buf
    
    def close(self):
        """環境のクリーンアップ"""
        pass

# # 使用例とデモンストレーション
# def demo_environment():
#     """環境のデモンストレーション"""
#     print("DataFrameSequentialEnv のデモンストレーション")
#     print("=" * 50)
    
#     # 環境の作成
#     env = DataFrameSequentialEnv(window_size=5, max_steps=20, render_mode='human')
    
#     # データ情報の表示
#     print(f"データ形状: {env.data.shape}")
#     print(f"特徴量: {list(env.data.columns)}")
#     print(f"観測空間: {env.observation_space}")
#     print(f"行動空間: {env.action_space}")
#     print()
    
#     # エピソードの実行
#     observation, info = env.reset()
#     print(f"初期観測: {observation[:5]}...")  # 最初の5要素のみ表示
#     print(f"初期情報: {info}")
    
#     total_reward = 0
#     for step in range(10):
#         # ランダム行動
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)
        
#         total_reward += reward
        
#         action_names = ['Hold', 'Buy', 'Sell']
#         print(f"ステップ {step + 1}: 行動={action_names[action]}, 報酬={reward:.3f}, 累積報酬={total_reward:.3f}")
        
#         if terminated or truncated:
#             break
    
#     print(f"\n最終累積報酬: {total_reward:.3f}")
    
#     # 描画
#     env.render()
    
#     env.close()

# # カスタムデータでの環境作成例
# def create_custom_data_env():
#     """カスタムデータを使用した環境作成例"""
#     # カスタムデータの作成
#     np.random.seed(123)
#     dates = pd.date_range('2024-01-01', periods=200, freq='H')
    
#     custom_data = pd.DataFrame({
#         'timestamp': dates,
#         'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(200) / 24) + np.random.normal(0, 2, 200),
#         'humidity': 50 + 20 * np.sin(2 * np.pi * np.arange(200) / 24 + np.pi/4) + np.random.normal(0, 5, 200),
#         'pressure': 1013 + np.random.normal(0, 10, 200),
#         'wind_speed': np.abs(np.random.normal(5, 3, 200))
#     })
    
#     # カスタム環境の作成
#     custom_env = DataFrameSequentialEnv(
#         data=custom_data,
#         window_size=12,  # 12時間のウィンドウ
#         max_steps=50
#     )
    
#     print("カスタムデータ環境の情報:")
#     print(f"データ形状: {custom_env.data.shape}")
#     print(f"特徴量: {list(custom_env.data.columns)}")
#     print(f"データ範囲: {custom_env.data['timestamp'].min()} - {custom_env.data['timestamp'].max()}")
    
#     return custom_env

# if __name__ == "__main__":
#     # デモの実行
#     demo_environment()
    
#     # カスタムデータ環境の作成例
#     print("\n" + "=" * 50)
#     custom_env = create_custom_data_env()