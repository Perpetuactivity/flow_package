import gymnasium as gym
import numpy as np
import pandas as pd


from dataclasses import dataclass
from gymnasium.vector import VectorWrapper
"""
reward_list: A list of rewards corresponding to different actions.
ex) row: predict, column: actual (answer)

            ["action_0",    "action_1",    "action_2", "..."]
"action_0"  [  1,         3,        3,       ... ]
"action_1"  [ 5,          2,         4,       ... ]
"action_2"  [ 5,         4,          2,        ... ]
...

---- Summary ----
X, Y: upper 0, X != Y

| p | a | R |
|---|---|---|
| 0 | 0 | 1 | [0] |
| X | X | 2 | [1] |
| 0 | X | 3 | [2] |
| X | Y | 4 | [3] |
| X | 0 | 5 | [4] |
"""
@dataclass
class EnvConfig:
    df_data: pd.DataFrame
    label_column: str
    reward_list: list
    max_steps: int
    normalize_method: str = 'min-max'
    rolling_window: int = 5
    test_mode: bool = False
    n_actions: int = None


def _check_input(config: EnvConfig):
    # Validate the input configuration
    if config.label_column not in config.df_data.columns:
        raise ValueError(f"Label column '{config.label_column}' not found in DataFrame columns.")
    if not isinstance(config.reward_list, list) or len(config.reward_list) == 0:
        raise ValueError("Reward list must be a non-empty list.")
    if config.normalize_method not in ['min-max', 'z-score', 'none']:
        raise ValueError("Normalization method must be one of 'min-max', 'z-score', or 'none'.")
    if config.rolling_window <= 0:
        raise ValueError("Rolling window must be a positive integer.")
    if config.max_steps <= 0:
        raise ValueError("Max steps must be a positive integer.")
    if config.max_steps < config.rolling_window:
        raise ValueError("Max steps must be greater than or equal to rolling window.")
    if config.df_data.empty:
        raise ValueError("DataFrame cannot be empty.")
    elif len(config.df_data) < config.max_steps:
        raise ValueError("DataFrame length must be greater than or equal to max_steps.")
    
    return


def _min_max_scalar(array):
    min_val = np.min(array)
    max_val = np.max(array)
    if max_val == min_val:
        return 0.0
    # Return the normalized value of the last element in the window
    return (array[-1] - min_val) / (max_val - min_val)

def _normalize(data: pd.DataFrame, method: str, rolling_window: int, label_column: str) -> pd.DataFrame:
    # Reset index to avoid duplicate index issues
    data_reset = data.reset_index(drop=True)
    normalized_data = data_reset.copy()
    
    for col in data_reset.columns:
        if col == label_column:  # Use the actual label column name
            continue
        if method == 'min-max':
            normalized_data[col] = data_reset[col].rolling(rolling_window).apply(_min_max_scalar, raw=True)
            # Fix: Use proper .iloc indexing to avoid duplicate index issues
            if rolling_window > 1 and len(normalized_data) > rolling_window - 1:
                normalized_data.iloc[:rolling_window-1, normalized_data.columns.get_loc(col)] = normalized_data.iloc[rolling_window-1, normalized_data.columns.get_loc(col)]
        elif method == 'z-score':
            normalized_data[col] = (data_reset[col] - data_reset[col].mean()) / data_reset[col].std()
        elif method == 'none':
            normalized_data[col] = data_reset[col]
        else:
            raise ValueError("Normalization method must be one of 'min-max', 'z-score', or 'none'.")
        if normalized_data[col].isna().values.any():
            print(f"Warning: NaN values found in column '{col}' after normalization.")
    
    if label_column not in normalized_data.columns:
        raise ValueError(f"Label column '{label_column}' not found in DataFrame after normalization.")
    return normalized_data.astype(np.float32)


class MultiDfEnvV2(gym.Env):
    def __init__(self, config: EnvConfig):
        super(MultiDfEnvV2, self).__init__()
        _check_input(config)

        self.original_data = config.df_data
        self.label_column = config.label_column

        self.reward_list = config.reward_list
        self.max_steps = config.max_steps
        self.normalize_method = config.normalize_method
        self.rolling_window = config.rolling_window
        self.test_mode = config.test_mode
        
        self.n_features = len(self.original_data.columns) - 1  # Exclude label column
        
        self.action_space = gym.spaces.Discrete(
            len(self.original_data[self.label_column].unique()) if config.n_actions is None else config.n_actions
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # Additional initialization
        self.current_step = 0
        self.normalized_data = pd.DataFrame()
        self.episode_count = 0
    
    def _choice_data(self):
        if self.test_mode or len(self.original_data) <= self.max_steps:
            self.normalized_data = _normalize(
                self.original_data.copy(),
                self.normalize_method,
                self.rolling_window,
                self.label_column  # Pass the actual label column name
            )
            return self.normalized_data
        
        start_idx = np.random.randint(0, len(self.original_data) - self.rolling_window)
        end_limit = min(len(self.original_data), start_idx + self.max_steps)
        end_idx = np.random.randint(start_idx + self.rolling_window, end_limit + 1)

        picked_data = self.original_data.iloc[start_idx:end_idx].reset_index(drop=True)
        self.normalized_data = _normalize(
            picked_data.copy(),
            self.normalize_method,
            self.rolling_window,
            self.label_column  # Pass the actual label column name
        )
        return self.normalized_data


    def _action_to_reward(self, action, actual):
        # print(f"Action taken: {action}, Actual label: {actual}")
        if action == 0:
            if actual == 0:
                return self.reward_list[0]
            else:
                return self.reward_list[2]
        else:
            if actual == 0:
                return self.reward_list[4]
            elif action == actual:
                return self.reward_list[1]
            else:
                return self.reward_list[3]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        self.normalized_data = self._choice_data()
        if "Label" not in self.normalized_data.columns:
            raise ValueError("Label column not found in DataFrame after normalization.")

        info = {
            "episode": self.episode_count,
            "steps": self.current_step,
            "data_length": len(self.normalized_data)
        }
        
        observation = self.normalized_data.drop(columns=[self.label_column]).iloc[self.current_step]
        return observation.values, info

    def step(self, action):
        # Apply action and return the new state, reward, done, and info
        current_label = self.normalized_data[self.label_column].iloc[self.current_step]

        self.current_step += 1

        done = self.current_step >= len(self.normalized_data)
        if done:
            # 最後のステップでも現在の観測値を返す（またはNone）
            observation = None  # または self.normalized_data.drop(columns=[self.label_column]).iloc[self.current_step-1]
        else:
            observation = self.normalized_data.drop(columns=[self.label_column]).iloc[self.current_step]
        reward = self._action_to_reward(action, current_label)

        truncated = False

        info = {
            "episode": self.episode_count,
            "steps": self.current_step,
            "data_length": len(self.normalized_data),
            "matrix_position": (action, current_label)
        }

        return observation, reward, done, truncated, info


class TestEnvWrapper(VectorWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.venv = venv
        self.finished_envs = np.zeros(venv.num_envs, dtype=np.bool)
    
    def step(self, actions):
        obs, rewards, dones, truncated, infos = self.venv.step(actions)


        # 終了していない環境のインデックスを取得
        active_envs = ~self.finished_envs
        
        # 終了していない環境のみのデータを返す
        """
        {
            "episode": self.episode_count,
            "steps": self.current_step,
            "data_length": len(self.normalized_data),
            "matrix_position": (action, current_label)
        }
        """
        self.finished_envs |= dones
        # infoの各要素を配列として処理
        filtered_infos = infos.copy() if infos else {}
        
        # matrix_positionのみをフィルタリング
        if infos and isinstance(infos, dict) and "matrix_position" in infos:
            matrix_positions = infos["matrix_position"]
            if isinstance(matrix_positions, (list, np.ndarray)):
                # アクティブな環境のmatrix_positionのみを抽出
                filtered_infos["matrix_position"] = np.array(matrix_positions)[active_envs]
        
        
        return obs, rewards, dones, truncated, filtered_infos
    
    def reset(self, **kwargs):
        self.finished_envs.fill(False)
        return self.venv.reset(**kwargs)
    