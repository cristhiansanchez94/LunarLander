from gym.envs.registration import register

register(
    id='LunarLauncher-v0',
    entry_point='LunarLauncher.envs:LunarLauncher',
)