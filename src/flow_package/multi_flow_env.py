from gymnasium import spaces
import gymnasium as gym
import numpy as np

from .preprocessing import normalization

class InputType:
    def __init__(
        self,
        data,
        sample_size=1000,
        is_test=False,
        normalize_exclude_columns=[],
        exclude_columns=[],
        reward_list=[1.0, -1.0]
    ):
        self.data = data
        self.sample_size = sample_size
        self.is_test = is_test
        self.normalize_exclude_columns = normalize_exclude_columns
        self.exclude_columns = exclude_columns
        self.reward_list = reward_list


class MultiFlowEnv(gym.Env):
    def __init__(
        self,
        input_type: InputType
    ):
        super(MultiFlowEnv, self).__init__()

        self.data = input_type.data
        self.sample_size = input_type.sample_size
        self.is_test = input_type.is_test
        self.normalize_exclude_columns = input_type.normalize_exclude_columns
        self.exclude_columns = input_type.exclude_columns + ["Label"]
        self.reward_list = input_type.reward_list
        self.action_space = spaces.Discrete(self.data["Label"].unique().size)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.data.columns) - len(self.exclude_columns),),
            dtype=np.float32
        )

        self.sample_df = None
        self.index = 0
    
    def reset(self):
        if self.is_test:
            self.sample_df = {
                "features": self.data.drop(columns=self.exclude_columns),
                "labels": self.data["Label"]
            }
        else:
            buf = self.data.sample(n=self.sample_size)
            self.sample_df = {
                "features": buf.drop(columns=self.exclude_columns),
                "labels": buf["Label"]
            }
        self.sample_df["features"] = normalization(
            self.sample_df["features"],
            categorical_columns=self.normalize_exclude_columns
        )
        self.index = 0

        return self.sample_df["features"].iloc[self.index].values
    
    def step(self, action):
        answer = self.sample_df["labels"].iloc[self.index]

        if self.index == self.sample_size - 1:
            terminated = True
            observation = None
        else:
            terminated = False
            self.index += 1
            observation = self.sample_df["features"].iloc[self.index].values

        reward = self.reward_list[int(action == answer)]

        """
        | action\\answer | 0 | other | true |
        | ------------- | --- | --- | --- |
        | 0 | TN | FP | FN |
        | other | FP? | x | FP? |
        | true | FP? | FP? | TP |

        [
            [TN, FP, FN],
            [FP, x, FP],
            [FN, FP, TP]
        ]
        """

        info = {
            "matrix_position": (action, answer),
            "action": action,
            "answer": answer
        }

        return observation, reward, terminated, False, info
    
    def render(self, mode="human"):
        pass
    
    def close(self):
        pass
