from gymnasium.envs.registration import register

from .const import *
from .preprocessing import *
from .utils import *

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

register(
    id='MultiFlow-v2',
    entry_point='flow_package:MultiDfEnv',
)

register(
    id='MultiFlow-v3',
    entry_point='flow_package:MultiDfEnvV2',
)

