import copy

#! /usr/bin/env python
# import dmcgym
import jax.profiler
import jax
# timer
import time

import gym
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags
import numpy as np

import dmc2gym

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


from jaxrl5.agents import ScoreMatchingLearner, TD3Learner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import wrap_gym
from jaxrl5.wrappers.wandb_video import WANDBVideo


def parse_hps_to_file():
    source_file_name = "configs/score_matching_config.py"
    target_file_name = "configs/tmp_score_matching_config.py"
    # Load python file
    with open(source_file_name, "r") as file:
        config_data = copy.deepcopy(file.readlines())
        # Replace hyperparameters
        # Denoising steps
        config_data[9] = config_data[9].replace("T = 5", f"T = {denoising_steps}")
        # Alpha value
        config_data[10] = config_data[10].replace("M_q = 50", f"M_q = {alpha}")
        # Critic hidden, turn width and depth into tuple
        crit_hidden_str = f"({','.join([str(critic_width)] * critic_depth)})"
        config_data[12] = config_data[12].replace("critic_hidden_dims=(512,512)",
                                f"critic_hidden_dims={crit_hidden_str}")
        # Actor hidden, turn width and depth into tuple
        act_hidden_str = f"({','.join([str(actor_width)] * actor_depth)})"
        config_data[13] = config_data[13].replace("actor_hidden_dims=(512,512)",
                                f"actor_hidden_dims={act_hidden_str}")
        # Write to new file
        with open(target_file_name, "w") as target_file:
            target_file.writelines(config_data)


FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "jaxrl5_online", "wandb project name.")
flags.DEFINE_string("run_name", "", "wandb run name.")
flags.DEFINE_string("env_name", "cartpole_balance", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer(
    "start_training", int(1e4), "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("wandb", True, "Use wandb")
flags.DEFINE_boolean("no_reset_env", False, "Turn off environment resets")
flags.DEFINE_boolean("save_video", True, "Save videos during evaluation.")
flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

# Config training hyperparameters
denoising_steps = 10
alpha = 100
actor_width = 1024
actor_depth = 3
critic_width = 1024
critic_depth = 3


flags.DEFINE_integer("T", denoising_steps, "Denoising steps.")
flags.DEFINE_integer("M_q", alpha, "Alpha value.")

flags.DEFINE_integer("actor_hidden", actor_width, "Width of policy net.")
flags.DEFINE_integer("actor_depth", actor_depth, "Depth of policy net.")

flags.DEFINE_integer("critic_hidden", critic_width, "Width of critic net.")
flags.DEFINE_integer("critic_depth", critic_depth, "Depth of critic net.")

parse_hps_to_file()

config_flags.DEFINE_config_file(
    "config",
    "configs/score_matching_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    if FLAGS.wandb:
        if FLAGS.run_name != "":
            wandb.init(project=FLAGS.project_name, name=FLAGS.run_name, tags=[FLAGS.run_name])
        else:
            wandb.init(project=FLAGS.project_name)
        wandb.config.update(FLAGS)

    suite, task = FLAGS.env_name.split('_')
    env = dmc2gym.make(domain_name=suite, task_name=task, seed=1)
    # env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    # note that video is currently crazy when we turn off environment resets
    if FLAGS.wandb and FLAGS.save_video and not FLAGS.no_reset_env:
        env = WANDBVideo(env)
    env.seed(FLAGS.seed)

    print('Evironment info:')
    print(env.observation_space)
    print(env.action_space)

    eval_env = dmc2gym.make(domain_name=suite, task_name=task, seed=1)
    # eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, env.action_space, **kwargs
    )

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    observation, done = env.reset(), False


    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):
        if i < FLAGS.start_training:
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
            if not FLAGS.no_reset_env:
                observation, done = env.reset(), False

                if FLAGS.wandb:
                    for k, v in info["episode"].items():
                        decode = {"r": "return", "l": "length", "t": "time"}
                        wandb.log({f"training/{decode[k]}": v}, step=i)
            else:
                done = False

                if FLAGS.wandb and i % FLAGS.log_interval == 0:
                    for k, v in info["episode"].items():
                        decode = {"r": "return", "l": "length", "t": "time"}
                        wandb.log({f"training/{decode[k]}": v}, step=i)
            

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size * FLAGS.utd_ratio)
            agent, update_info = agent.update(batch)



            if FLAGS.wandb and i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"training/{k}": v}, step=i)

        if FLAGS.wandb and i % FLAGS.eval_interval == 0:
            eval_info = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
                save_video=FLAGS.save_video,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=i)

if __name__ == "__main__":
    app.run(main)
