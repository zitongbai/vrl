from legged_gym.envs import PerceptiveRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .go2_perceptive_config import Go2PerceptiveCfg

class Go2Perceptive(PerceptiveRobot):
    """
    Go2 with perceptive capabilities.
    """
    cfg : Go2PerceptiveCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        pass
        
        