from os import listdir
from pathlib import Path

import json
from os.path import join
import argparse
import gym
import numpy as np

import highway_env_local.envs
import os
import sys
from highway_env_local.envs import highway_env_local
from multi_head.DQNAgent_local.evaluation_local import Evaluation, logger
from rl_agents.agents.common.exploration.abstract import exploration_factory
# from utils import show_videos
#from highway_env_local.scripts.utils import show_videos
from rl_agents.agents.common.factory import agent_factory

# agent_test = "configs/HighwayEnv/agents/DQNAgent/agent_metadata.json"
# env = load_environment(env_config)
# env.configure({"offscreen_rendering": True})
# env.reset()
# #yaeli
# agent = load_agent(agent_config, env)
# agent_1 = load_agent(agent_test, env)
# evaluation = Evaluation(env, agent, num_episodes=3000, recover=True)
# evaluation.test()
# # show_videos(evaluation.run_directory)

class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='out', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes,
                                           display_env=display_env)



def config(env_config, agent_config):
    env = gym.make(env_config["id"])
    env.configure(env_config)
    env.define_spaces()
    agent = agent_factory(env, agent_config)
    return env, agent


def train_agent(env_config_path, agent_config_path, num_episodes):
    """train agent"""
    f1, f2 = open(env_config_path), open(agent_config_path)
    env_config, agent_config = json.load(f1), json.load(f2)
    env, agent = config(env_config, agent_config)
    evaluation = MyEvaluation(env, agent, output_dir='out', num_episodes=num_episodes,
                              display_env=False)
    evaluation.train()
    return evaluation


def load_evaluation_agent(load_path, num_episodes):
    """load agent"""
    config_filename = [x for x in listdir(load_path) if "metadata" in x][0]
    f = open(join(load_path, config_filename))
    config_dict = json.load(f)
    env_config, agent_config, = config_dict['env'], config_dict['agent']
    env, agent = config(env_config, agent_config)
    agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
    evaluation = MyEvaluation(env, agent, num_episodes=num_episodes, display_env=True)
    agent_path = Path(join(load_path, 'checkpoint-final.tar'))
    evaluation.load_agent_model(agent_path)
    return evaluation

def test_agent(evaluation):
    evaluation.test()

def main(args):
    evaluation = load_evaluation_agent(args.load_path, args.num_episodes) if args.load_path \
        else train_agent(args.env_config, args.agent_config, args.num_episodes)
    if args.eval: test_agent(evaluation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='multi head')
    args = parser.parse_args()

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

    args.env_config = 'configs/my_env_config.json'
    args.agent_config = 'configs/ddqn_agent.json'
    args.num_episodes = 3
    args.load_path = '/home/yotama/OneDrive/Local_Git/Multi_Head/multi_head/out/HighwayEnvLocal/DQNAgent/run_20210825-112722_81785'
    args.eval = True

    main(args)