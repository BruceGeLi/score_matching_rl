import copy
import logging
import sys
from datetime import datetime
from typing import Union

# ! /usr/bin/env python
# import dmcgym
import jax.profiler
import jax
# timer
import time

import gym
import psutil
import tqdm
import wandb
import numpy as np

import dmc2gym

import os
from pathlib import Path

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from jaxrl5.agents import ScoreMatchingLearner, TD3Learner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo

from cw2 import cluster_work
from cw2 import cw_error
from cw2 import experiment
from cw2.cw_data import cw_logging
from configs.score_matching_config import get_config


def parse_train_hps(denoising_steps, alpha, critic_width, critic_depth,
                    actor_width, actor_depth):
    train_configs = get_config()

    # Denoising steps
    train_configs.T = denoising_steps
    # Alpha value
    train_configs.M_q = alpha
    # Critic hidden, turn width and depth into tuple
    train_configs.critic_hidden_dims = (critic_width,) * critic_depth
    # Actor hidden, turn width and depth into tuple
    train_configs.actor_hidden_dims = (actor_width,) * actor_depth
    return train_configs


def is_slurm(cw: cluster_work.ClusterWork):
    if cw.args["slurm"]:
        return True
    else:
        return False


def is_on_local_machine():
    if any(["local" in argv for argv in sys.argv]):
        return True
    else:
        return False


def get_formatted_date_time() -> str:
    """
    Get formatted date and time, e.g. May-01-2021 22:14:31
    Returns:
        dt_string: date time string
    """
    now = datetime.now()
    dt_string = now.strftime("%b-%2d-%Y-%H:%M:%S")
    return dt_string


def join_path(*paths: Union[str]) -> str:
    """

    Args:
        *paths: paths to join

    Returns:
        joined path
    """
    return os.path.join(*paths)


def mkdir(directory: str, overwrite: bool = False):
    """

    Args:
        directory: dir path to make
        overwrite: overwrite exist dir

    Returns:
        None

    Raise:
        FileExistsError if dir exists and overwrite is False
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=overwrite)
    except FileExistsError:
        logging.error("Directory already exists, remove it before make a new one.")
        raise


def dir_go_up(num_level: int = 2, current_file_dir: str = "default") -> str:
    """
    Go to upper n level of current file directory
    Args:
        num_level: number of level to go up
        current_file_dir: current dir

    Returns:
        dir n level up
    """
    if current_file_dir == "default":
        current_file_dir = os.path.realpath(__file__)
    while num_level != 0:
        current_file_dir = os.path.dirname(current_file_dir)
        num_level -= 1
    return current_file_dir


def make_log_dir_with_time_stamp(log_name: str) -> str:
    """
    Get the dir to the log
    Args:
        log_name: log's name

    Returns:
        directory to log file
    """

    return os.path.join(dir_go_up(2), "log", log_name,
                        get_formatted_date_time())


def set_value_in_nest_dict(config, key, value):
    """
    Set value of a certain key in a recursive way in a nested dictionary

    Args:
        config: configuration dictionary
        key: key to ref
        value: value to set

    Returns:
        config
    """
    for k in config.keys():
        if k == key:
            config[k] = value
        if isinstance(config[k], dict):
            set_value_in_nest_dict(config[k], key, value)
    return config


def process_train_rep_config_file(cw, config_obj):
    """
    Given processed cw2 configuration, do further process, including:
    - Overwrite log path with time stamp
    - Create model save folders
    - Overwrite random seed by the repetition number
    - Save the current repository commits
    - Make a copy of the config and restore the exp path to the original
    - Dump this copied config into yaml file into the model save folder
    - Dump the current time stamped config file in log folder to make slurm
      call bug free
    Args:
        exp_configs: list of configs processed by cw2 already

    Returns:
        None

    """
    exp_configs = config_obj.exp_configs
    formatted_time = get_formatted_date_time()
    # Loop over the config of each repetition
    for i, rep_config in enumerate(exp_configs):
        # overwrite the log path in case of code copy only in slurm
        if (is_slurm(cw) and "experiment_copy_auto_dst"
                in config_obj.slurm_config.keys()):
            # self.manage_code_copy_path(rep_config)
            pass

        # Add time stamp to log directory
        log_path = os.path.abspath(rep_config["log_path"])
        rep_log_path = os.path.abspath(rep_config["_rep_log_path"])
        rep_config["log_path"] = \
            log_path.replace("log", f"log_{formatted_time}")
        rep_config["_rep_log_path"] = \
            rep_log_path.replace("log", f"log_{formatted_time}")

        # Make model save directory
        model_save_dir = join_path(rep_config["_rep_log_path"],
                                   "model")
        try:
            mkdir(os.path.abspath(model_save_dir))
        except FileExistsError:
            import logging
            logging.error(formatted_time)
            raise

        # Set random seed to the repetition number
        set_value_in_nest_dict(rep_config, "seed",
                               rep_config['_rep_idx'])

        # Make a hard copy of the config
        copied_rep_config = copy.deepcopy(rep_config)

        # Recover the path to its original
        copied_rep_config["path"] = copied_rep_config["_basic_path"]

        # Reset the repetition number to 1 for future test usage
        copied_rep_config["repetitions"] = 1
        if copied_rep_config.get("reps_in_parallel", False):
            del copied_rep_config["reps_in_parallel"]
        if copied_rep_config.get("reps_per_job", False):
            del copied_rep_config["reps_per_job"]

        # Delete the generated cw2 configs
        for key in rep_config.keys():
            if key[0] == "_":
                del copied_rep_config[key]
        del copied_rep_config["log_path"]

    # Save the time stamped config file in local /log directory
    time_stamped_config_path = make_log_dir_with_time_stamp("")
    mkdir(time_stamped_config_path, overwrite=True)

    config_obj.to_yaml(time_stamped_config_path,
                       relpath=False)
    config_obj.config_path = join_path(time_stamped_config_path,
                                       "relative_" + config_obj.f_name)


class QSMExperiment(experiment.AbstractExperiment):
    def initialize(self, cw_config: dict, rep: int,
                   logger: cw_logging.LoggerArray) -> None:

        # Get experiment config
        cfg = cw_config["params"]
        # cpu_cores = cw_config.get("cpu_cores", None)
        # if cpu_cores is None:
        #     cpu_cores = set(range(psutil.cpu_count(logical=True)))

        # Note, initialization of the official code
        project_name = cfg["wandb"]["project"]
        group_name = cfg["wandb"]["group"]
        wandb_run_name = ""  # "wandb run name."
        env_name = cfg["env_id"]
        seed = rep
        # seed = 42
        eval_episodes = 1
        log_interval = 1000
        eval_interval = 10000
        batch_size = 256
        max_steps = int(1e6)
        # max_steps = 15000
        start_training = int(1e4)
        use_tqdm = True
        use_wandb = True
        no_reset_env = False
        save_video = False
        utd_ratio = 1

        # Config training hyperparameters
        denoising_steps = cfg["denoising_steps"]
        alpha = cfg["alpha"]
        actor_width = cfg["actor_width"]
        actor_depth = cfg["actor_depth"]
        critic_width = cfg["critic_width"]
        critic_depth = cfg["critic_depth"]

        train_configs = parse_train_hps(denoising_steps, alpha, critic_width,
                                        critic_depth, actor_width,
                                        actor_depth)

        if use_wandb:
            if wandb_run_name != "":
                self.wandb_run = wandb.init(project=project_name,
                                            name=wandb_run_name,
                                            group=group_name,
                                            tags=[wandb_run_name])
            else:
                self.wandb_run = wandb.init(project=project_name,
                                            group=group_name, )

        self.wandb_run.config.update(dict(train_configs))
        self.wandb_run.config.update({"seed": seed})

        suite, task = env_name.split('_')
        print(f"Env name: {env_name}")
        env = dmc2gym.make(domain_name=suite, task_name=task, seed=1)
        # env = gym.make(env_name)

        env = wrap_gym(env, rescale_actions=True)
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
        # note that video is currently crazy when we turn off environment resets
        if use_wandb and save_video and not no_reset_env:
            env = WANDBVideo(env)
        env.seed(seed)

        print('Evironment info:')
        print(env.observation_space)
        print(env.action_space)

        eval_env = dmc2gym.make(domain_name=suite, task_name=task, seed=1)
        # eval_env = gym.make(env_name)
        eval_env = wrap_gym(eval_env, rescale_actions=True)
        eval_env.seed(seed + 42)

        kwargs = dict(train_configs)
        model_cls = kwargs.pop("model_cls")
        agent = globals()[model_cls].create(
            seed, env.observation_space, env.action_space, **kwargs
        )

        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, max_steps
        )
        replay_buffer.seed(seed)

        observation, done = env.reset(), False

        for i in tqdm.tqdm(
                range(1, max_steps + 1), smoothing=0.1, disable=not use_tqdm
        ):
            if i < start_training:
                action = env.action_space.sample()
            else:
                # print(observation)
                action, agent = agent.sample_actions(observation)
                action = np.asarray(action)
            next_observation, reward, done, info = env.step(action)

            if not done or "TimeLimit.truncated" in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(
                dict(
                    observations=observation,
                    actions=action,
                    rewards=reward,
                    masks=mask,
                    dones=done,
                    next_observations=next_observation,
                )
            )
            observation = next_observation

            if done:
                if not no_reset_env:
                    observation, done = env.reset(), False

                    if use_wandb:
                        for k, v in info["episode"].items():
                            decode = {"r": "return", "l": "length", "t": "time"}
                            self.wandb_run.log({f"training/{decode[k]}": v},
                                               step=i)
                else:
                    done = False

                    if use_wandb and i % log_interval == 0:
                        for k, v in info["episode"].items():
                            decode = {"r": "return", "l": "length", "t": "time"}
                            self.wandb_run.log({f"training/{decode[k]}": v},
                                               step=i)

            if i >= start_training:
                batch = replay_buffer.sample(batch_size * utd_ratio)
                agent, update_info = agent.update(batch)

                if use_wandb and i % log_interval == 0:
                    for k, v in update_info.items():
                        self.wandb_run.log({f"training/{k}": v}, step=i)

            if use_wandb and i % eval_interval == 0:
                eval_info = evaluate(
                    agent,
                    eval_env,
                    num_episodes=eval_episodes,
                    save_video=save_video,
                )
                for k, v in eval_info.items():
                    self.wandb_run.log({f"evaluation/{k}": v}, step=i)

    def run(self, config: dict, rep: int,
            logger: cw_logging.LoggerArray) -> None:
        pass

    def finalize(
            self, surrender: cw_error.ExperimentSurrender = None,
            crash: bool = False
    ):
        if surrender is not None:
            cw_logging.getLogger().info("Run was surrendered early.")

        if crash:
            cw_logging.getLogger().warning("Run crashed with an exception.")
        cw_logging.getLogger().info("Finished. Closing Down.")
        self.wandb_run.finish()


if __name__ == "__main__":
    for key in os.environ.keys():
        if "-xCORE-AVX2" in os.environ[key]:
            os.environ[key] = os.environ[key].replace("-xCORE-AVX2", "")

    cw = cluster_work.ClusterWork(QSMExperiment)
    if is_slurm(cw) or is_on_local_machine():
        process_train_rep_config_file(cw, cw.config)
    cw.run()
