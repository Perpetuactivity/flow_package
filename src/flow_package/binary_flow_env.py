import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class InputType:
    def __init__(
            self,
            input_features,
            input_labels,
            normal_label,
            reward_list
    ):
        if len(reward_list) != 2:
            raise ValueError("The length of the reward_list is not the same as the number of labels")
        
        self.input_features = input_features
        self.input_labels = input_labels
        self.normal_label = normal_label
        self.reward_list = reward_list

class BinaryFlowEnv(gym.Env):
    def __init__(self, input_type: InputType):
        super(BinaryFlowEnv, self).__init__()

        self.input_features = input_type.input_features
        self.input_labels = input_type.input_labels
        self.normal_label = input_type.normal_label
        self.reward_list = input_type.reward_list

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.input_features.columns),), dtype=np.float32
        )

        self.rng = np.random.default_rng(0)

        self.state = {}
        self.data_len = len(self.input_features)
        self.index_array = np.arange(0, self.data_len - 1)
        self.index = self.rng.choice(self.index_array, 1)[0]

    def reset(self):
        super().reset()

        self.state = {}
        self.data_len = len(self.input_features)
        self.index_array = np.arange(0, self.data_len - 1)
        self.index = self.rng.choice(self.index_array, 1)[0]

        np.delete(self.index_array, self.index)
        self.state = self.input_features.iloc[self.index].values

        return self.state
    
    def step(self, action):
        answer = self.input_labels.iloc[self.index]

        if answer == self.normal_label:
            answer = 0
        else:
            answer = 1
        
        self.index = self.rng.choice(self.index_array, 1)[0]

        reward = self.reward_list[0] if action == answer else self.reward_list[1]

        # TP, FP, TN, FN
        if action == answer:
            if action == 1:
                matrix_position = (1, 1)
            else:
                matrix_position = (0, 0)
        else:
            if action == 1:
                matrix_position = (1, 0)
            else:
                matrix_position = (0, 1)
        
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
        
        terminated = random.random() < 0.01

        return observation, reward, terminated, False, info
    
    def render(self, mode="human"):
        pass

    def close(self):
        pass
