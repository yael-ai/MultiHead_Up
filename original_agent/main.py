from os import listdir
from pathlib import Path

import json
from os.path import join
import argparse
import gym
import numpy as np
import os
import sys

from rl_agents.agents.common.exploration.abstract import exploration_factory
# from utils import show_videos
# from highway_env_local.scripts.utils import show_videos
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation
from original_agent.highway_env_original import *


def config(env_config, agent_config):
    env = gym.make(env_config["id"])
    env.configure(env_config)
    env.define_spaces()
    agent = agent_factory(env, agent_config)
    return env, agent


def train_agent(env_config_path, agent_config_path, num_episodes, dispaly_env):
    """train agent"""
    f1, f2 = open(env_config_path), open(agent_config_path)
    env_config, agent_config = json.load(f1), json.load(f2)
    env, agent = config(env_config,agent_config)
    evaluation = Evaluation(env, agent, num_episodes=num_episodes, display_env=dispaly_env)
    evaluation.train()
    return evaluation


def load_evaluation_agent(load_path, num_episodes, dispaly_env):
    """load agent"""
    config_filename = [x for x in listdir(load_path) if "metadata" in x][0]
    f = open(join(load_path, config_filename))
    config_dict = json.load(f)
    env_config, agent_config, = config_dict['env'], config_dict['agent']
    env, agent = config(env_config, agent_config)
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    evaluation = Evaluation(env, agent, num_episodes=num_episodes, display_env=dispaly_env)
    agent_path = Path(join(load_path, 'checkpoint-final.tar'))
    evaluation.load_agent_model(agent_path)
    return evaluation


def test_agent(evaluation):
    evaluation.test()


def main(args):
    evaluation = load_evaluation_agent(args.load_path, args.num_episodes, args.dispaly_env)\
        if args.load_path \
        else train_agent(args.env_config, args.agent_config, args.num_episodes, args.dispaly_env)
    if args.eval: test_agent(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi head')
    args = parser.parse_args()
    args.load_path = None
    # if args.server:
    #     """server"""
    #     sys.path.insert(0,'/data/home/yael123/multi_head/highway_env_local/scripts/')
    #     rl_agents_dir = '/data/home/yael123/multi_head/rl-agents_local/'
    #     sys.path.append(rl_agents_dir)
    #     from highway_env_local.scripts.utils import show_videos
    # else:
    #     """local"""
    #     sys.path.insert(0, 'C:/Users/yael/Documents/GitHub/multi_head/highway_env_local/scripts/')
    #     rl_agents_dir = 'C:/Users/yael/Documents/GitHub/multi_head/rl-agents_local/'
    #     sys.path.append(rl_agents_dir)
    #     os.chdir(rl_agents_dir + "scripts")
    args.dispaly_env = True
    args.env_config = 'original_env_config.json'
    args.agent_config = 'ddqn_agent_original.json'
    args.num_episodes = 3
    args.load_path = 'out/HighwayEnvOriginal/DQNAgent/run_20210825-122306_99704'
    args.eval = True

    main(args)
