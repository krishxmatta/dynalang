import embodied

def get_explicit_spaces():
    obs_space = {
        "orientation": embodied.Space(dtype=np.uint8, shape=(4,), low=0, high=1),
        "pixels": embodied.Space(dtype=np.uint8, shape=(1920, 1080, 3), low=0, high=255),
        "timedelta": embodied.Space(dtype=np.int64, shape=(), low=-9223372036854775806, high=9223372036854775805),
        "reward": embodied.Space(dtype=np.float32, shape=(), low=-np.inf, high=np.inf),
        "is_first": embodied.Space(dtype=bool, shape=(), low=False, high=True),
        "is_last": embodied.Space(dtype=bool, shape=(), low=False, high=True),
        "is_terminal": embodied.Space(dtype=bool, shape=(), low=False, high=True)
    }

    act_space = {
        "action_type": embodied.Space(dtype=np.float32, shape=(2,), low=0, high=1),
        "touch_position": embodied.Space(dtype=np.float32, shape=(2,), low=-1.0, high=1.0),
        "reset": embodied.Space(dtype=bool, shape=(), low=False, high=True),
    }

    return obs_space, act_space

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
