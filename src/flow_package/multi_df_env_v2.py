import gymnasium as gym
import numpy as np
import pandas as pd


from dataclasses import dataclass

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


def _normalize(data: pd.DataFrame, method: str) -> pd.DataFrame:
    for col in data.columns:
        if col == 'Label':
            continue
        if method == 'min-max':
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        elif method == 'z-score':
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        elif method == 'none':
            continue
        else:
            raise ValueError("Normalization method must be one of 'min-max', 'z-score', or 'none'.")
    return data.astype(np.float32)


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
        
        self.action_space = gym.spaces.Discrete(len(self.original_data[self.label_column].unique()))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32
        )

        # Additional initialization
        self.current_step = 0
        self.normalized_data = pd.DataFrame()
        self.episode_count = 0
    
    def _choice_data(self):
        if self.test_mode or len(self.original_data) <= self.max_steps:
            return self.original_data
        
        start_idx = np.random.randint(0, len(self.original_data) - self.rolling_window)
        end_limit = min(len(self.original_data), start_idx + self.max_steps)
        end_idx = np.random.randint(start_idx + self.rolling_window, end_limit + 1)

        picked_data = self.original_data.iloc[start_idx:end_idx].reset_index(drop=True)
        self.normalized_data = _normalize(picked_data.copy(), self.normalize_method)
        return self.normalized_data


    def _action_to_reward(self, action, actual):
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
        self._choice_data()

        info = {
            "episode": self.episode_count,
            "steps": self.current_step
        }
        
        observation = self.normalized_data.drop(columns=[self.label_column]).iloc[self.current_step]
        return observation.values, info

    def step(self, action):
        # Apply action and return the new state, reward, done, and info
        current_label = self.normalized_data[self.label_column].iloc[self.current_step]

        self.current_step += 1

        observation = self.normalized_data.drop(columns=[self.label_column]).iloc[self.current_step]
        reward = self._action_to_reward(action, current_label)

        done = self.current_step >= len(self.normalized_data)
        truncated = False

        info = {
            "episode": self.episode_count,
            "steps": self.current_step
        }

        return observation.values, reward, done, truncated, info
