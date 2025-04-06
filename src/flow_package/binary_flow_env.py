import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

ERROR_LABEL = "The length of the reward_list is " \
    "not the same as the number of labels"


class InputType:
    def __init__(
            self,
            input_features,
            input_labels,
            # normal_label,
            reward_list,
            type_env=None
    ):
        if len(reward_list) != 2:
            raise ValueError(ERROR_LABEL)

        self.input_features = input_features
        self.input_labels = input_labels
        # self.normal_label = normal_label
        self.reward_list = reward_list
        self.type_env = type_env


class BinaryFlowEnv(gym.Env):
    def __init__(self, input_type: InputType):
        super(BinaryFlowEnv, self).__init__()

        self.input_features = input_type.input_features
        self.input_labels = input_type.input_labels
        # self.normal_label = input_type.normal_label
        self.reward_list = input_type.reward_list
        self.type_env = input_type.type_env

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.input_features.columns),),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(0)

        self.state = {}
        self.data_len = len(self.input_features)
        self.index_array = np.arange(self.data_len)
        if self.type_env is None:
            self.index = self.rng.choice(self.index_array, 1)[0]
        else:
            self.index = 0

    def reset(self):
        super().reset()

        self.state = {}
        self.data_len = len(self.input_features)
        self.index_array = np.arange(self.data_len)
        self.index = self.rng.choice(self.index_array, 1)[0]

        np.delete(self.index_array, self.index)
        self.state = self.input_features.iloc[self.index].values

        return self.state

    def step(self, action):
        answer = self.input_labels.iloc[self.index]

        # if answer == self.normal_label:
        #     answer = 0
        # else:
        #     answer = 1
        if self.type_env is None:
            self.index = self.rng.choice(self.index_array, 1)[0]
        else:
            self.index += 1

        reward = self.reward_list[int(action != answer)]

        # TP, FP, TN, FN
        matrix_position = (action, answer)

        info = {
            "matrix_position": matrix_position,
            "action": action,
            "answer": answer
        }

        try:
            observation = self.input_features.iloc[self.index].values
        except IndexError:
            self.index = 0
            observation = self.input_features.iloc[self.index].values

        if self.type_env is not None:
            terminated = self.index == 0
        else:
            terminated = random.random() < 0.01

        return observation, reward, terminated, False, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
