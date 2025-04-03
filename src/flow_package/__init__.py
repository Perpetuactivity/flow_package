from gymnasium.envs.registration import register

from .const import *
from .preprocessing import *
from .utils import *

from .binary_flow_env import *
from .multi_flow_env import *

# classify network flow as normal or attack
register(
    id='BinaryFlow-v1',
    entry_point='flow_package:BinaryFlowEnv',
)

# classify network flow as normal or attack(multi attack type)
register(
    id='MultipleFlow-v1',
    entry_point='flow_package:MultipleFlowEnv',
)

