from gymnasium.envs import register
import gymnasium as gym
from envs.ccpp.CCPP import CCPP_Env


register(id="CCPP-v0", entry_point="envs.ccpp:CCPP_Env")
register(id="CCPP-v1", entry_point="envs.ccpp:CCPP_Discrete")
# env = gym.make('CCPP-v0', agent_dims=[0.2, 0.2], agent_loc=[7,5, 0])
env = gym.make("CCPP-v1", agent_dims=[0.1, 0.1], agent_loc=[7.5, 0])
