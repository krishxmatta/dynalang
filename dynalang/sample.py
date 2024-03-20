import importlib
import pathlib
import numpy as np
import sys
import warnings
import os
from functools import partial as bind

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied
from embodied import wrappers
from embodied.core import path

def main(argv=None):
  from . import agent as agt

  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  config = config.update({"logdir": f"{config.logdir}_{config.seed}"})
  args = embodied.Config(
      **config.run, logdir=f"{config.logdir}",
      batch_steps=config.batch_size * config.batch_length)
  # print(config)

  logdir = embodied.Path(args.logdir)
  if args.script != 'parallel_env':
    logdir.mkdirs()
    config.save(logdir / 'config.yaml')
    step = embodied.Counter()
    logger = make_logger(parsed, logdir, step, config)

  cleanup = []
  try:
    if args.script == 'sample':
      replay = make_replay(config, logdir / 'episodes')
      env = wrapped_env(config, batch=True)
      cleanup.append(env)
      embodied.run.sample(env, replay, logger, args)
    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_logger(parsed, logdir, step, config):
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl',
                                  '(episode/score|real_step)'),
      embodied.logger.TensorBoardOutput(logdir),
  ], multiplier)
  if config.use_wandb:
    import wandb
    wandb_id_file = f"{str(logdir)}/wandb_id.txt"
    wandb_pa = path.Path(wandb_id_file)
    if wandb_pa.exists():
        print("!! Resuming wandb run !!")
        wandb_id = wandb_pa.read().strip()
    else:
        wandb_id = wandb.util.generate_id()
        wandb_pa.write(str(wandb_id))
    project = config.task
    wandb.init(
        id=wandb_id,
        resume="allow",
        project=project,
        name=logdir.name,
        group=logdir.name[:logdir.name.rfind("_")],
        sync_tensorboard=True,
        config=dict(config)
    )

  return logger


def make_replay(
    config, directory=None, is_eval=False, rate_limit=False,
    load_directories=None, **kwargs):
  assert config.replay == 'uniform' or not rate_limit
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if not (load_directories and load_directories[0]):
    load_directories = None
  if config.replay == 'uniform' or is_eval:
    kw = {
      'online': config.replay_online,
      'load_directories': load_directories,
      'dataset_excluded_keys': config.dataset_excluded_keys,
    }
    if rate_limit and config.run.train_ratio > 0:
      kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
      kw['tolerance'] = 10 * config.batch_size
      kw['min_size'] = config.batch_size
    replay = embodied.replay.Uniform(length, size, directory, **kw)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


def wrapped_env(config, batch, **overrides):
  ctor = bind(make_env, config, **overrides)
  if batch and config.envs.parallel != 'none':
    ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
  if config.envs.restart:
    ctor = bind(wrappers.RestartOnException, ctor)
  if batch:
    envs = [ctor() for _ in range(config.envs.amount)]
    return embodied.BatchEnv(envs, (config.envs.parallel != 'none'))
  else:
    return ctor()


def make_env(config, **overrides):
  from embodied.envs import from_gym
  suite, task = config.task.split('_', 1)
  ctor = {
    'dummy': 'embodied.envs.dummy:Dummy',
    'gym': 'embodied.envs.from_gym:FromGym',
    'dm': 'embodied.envs.from_dmenv:FromDM',
    'crafter': 'embodied.envs.crafter:Crafter',
    'dmc': 'embodied.envs.dmc:DMC',
    'atari': 'embodied.envs.atari:Atari',
    'atari100k': 'embodied.envs.atari:Atari',
    'dmlab': 'embodied.envs.dmlab:DMLab',
    'minecraft': 'embodied.envs.minecraft:Minecraft',
    'loconav': 'embodied.envs.loconav:LocoNav',
    'pinpad': 'embodied.envs.pinpad:PinPad',
    'messenger': 'embodied.envs.messenger:Messenger',
    'homegrid': 'embodied.envs.homegrid:HomeGrid',
    'vln': 'embodied.envs.vln:VLNEnv',
    'langroom': 'langroom:LangRoom',
    'procgen': lambda task, **kw: from_gym.FromGym(
        f'procgen:procgen-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  from embodied import wrappers
  args = config.wrapper
  # Env specific wrappers
  if hasattr(env, "wrappers"):
    for w in env.wrappers:
      env = w(env)

#  for name, space in env.obs_space.items():
#    if space.dtype in (np.uint32, np.uint64):
#      env = wrappers.OneHotObs(env, name)

  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif space.discrete:
      env = wrappers.OneHotAction(env, name)
    elif args.discretize:
      env = wrappers.DiscretizeAction(env, name, args.discretize)
    else:
      env = wrappers.NormalizeAction(env, name)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset, args.timeout_reward)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
