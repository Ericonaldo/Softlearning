import sys

from .ant import AntEnv
from .humanoid import HumanoidEnv
from .swimmer import SwimmerEnv

env_overwrite = {}#'Ant': AntEnv, 'Humanoid': HumanoidEnv, 'Swimmer':SwimmerEnv}

sys.modules[__name__] = env_overwrite
