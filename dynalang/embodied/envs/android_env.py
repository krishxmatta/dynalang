import embodied

class AndroidEnv(embodied.Env):
    def __init__(self,
                 task,
                 avd_name='avd',
                 android_avd_home='/Users/krishxmatta/.android/avd',
                 android_sdk_root='/Users/krishxmatta/Library/Android/sdk',
                 emulator_path='/Users/krishxmatta/Library/Android/sdk/emulator/emulator',
                 adb_path='/Users/krishxmatta/Library/Android/sdk/platform-tools/adb',
                 task_filename='mdp_0000.textproto'):
        from android_env import loader
        from android_env.components import config_classes
        from android_env.wrappers.gym_wrapper import GymInterfaceWrapper
        from . import from_gym

        task_path = f"android_env/tasks/{task}/{task_filename}"

        self._env = GymInterfaceWrapper(loader.load(
                                    avd_name='avd',
                                    android_avd_home='/Users/krishxmatta/.android/avd',
                                    android_sdk_root='/Users/krishxmatta/Library/Android/sdk',
                                    emulator_path='/Users/krishxmatta/Library/Android/sdk/emulator/emulator',
                                    adb_path='/Users/krishxmatta/Library/Android/sdk/platform-tools/adb',
                                    task_path='/Users/krishxmatta/Projects/androidenv/tasks/vokram/mdp_0000.textproto'))

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.wrappers = [from_gym.FromGym]

    def reset(self):
        obs = self._env.reset()
        obs["is_read_step"] = False
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs["is_read_step"] = False
        return obs, reward, done, info

    def render(self):
        return self._env.render()
