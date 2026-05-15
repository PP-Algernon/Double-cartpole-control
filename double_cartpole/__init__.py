"""Local source package hook for Gymnasium environment registration."""

from double_cartpole.double_cartpole_custom_gym_env import DoubleCartpoleEnv, ENV_ID, register_env

__all__ = ["DoubleCartpoleEnv", "ENV_ID", "register_env"]
