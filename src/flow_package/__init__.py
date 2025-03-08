from gymnasium.envs.registration import register

from .const import *
from .preprocessing import *

from .binary_flow_env import *
from .multi_flow_env import *

register(
    id='BinaryFlow-v1',
    entry_point='flow_package:BinaryFlowEnv',
)

register(
    id='MultipleFlow-v1',
    entry_point='flow_package:MultipleFlowEnv',
)

