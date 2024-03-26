import embodied
from browser_env import ScriptBrowserEnv
from . import from_gym

class WebArena(embodied.Env):
    def __init__(self, task, config_file="environments/webarena/config_files/examples/1.json", **kwargs):
        self._env = ScriptBrowserEnv(
            headless=True,
            observation_type = "accessibility_tree",
            **kwargs)
        self._config_file = config_file
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        _, _ = self._env.reset(options={"config_file": self._config_file})
        self.wrappers = [from_gym.FromGym]

    def reset(self):
        obs, _ = self.env.reset(options={"config_file": self._config_file})
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
