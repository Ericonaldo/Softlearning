import argparse
import importlib
import json
import os
from pathlib import Path
import pickle

import pandas as pd
import numpy as np

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.samplers import rollouts
from softlearning.utils.video import save_video
from runner import ExperimentRunner
from softlearning.misc.utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--load_epoch', type=str, default='final')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num_rollouts', '-n', type=int, default=1)
    # parser.add_argument('--store_traj', action='store_true')
    parser.add_argument('--render_mode', type=str, default=None)#rgb_array
    parser.add_argument('--video-save-path', type=Path, default=None)
    parser.add_argument('--traj-save-path', type=Path, default='./trajs/')

    args = parser.parse_args()

    return args

def load_environment(variant):
    environment_params = (
        variant['environment_params']['training']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    environment = get_environment_from_params(environment_params)
    return environment


def load_policy(checkpoint_dir, variant, environment):
    Qs = get_Q_function_from_variant(variant, environment)
    policy = get_policy_from_variant(variant, environment, Qs)

    policy_save_path = ExperimentRunner._policy_save_path(checkpoint_dir)
    status = policy.load_weights(policy_save_path)
    status.assert_consumed().run_restore_ops()

    policy.set_deterministic()

    return policy


def simulate_policy(checkpoint_dir,
                    num_rollouts,
                    max_path_length,
                    render_mode,
                    traj_save_path=None,
                    video_save_path=None,
                    variant=None,
                    **kwargs):
    checkpoint_dir = os.path.join(checkpoint_dir, variant['algorithm_params']['domain'])
    checkpoint_dir = os.path.join(checkpoint_dir, 'seed-{}'.format(args.seed))
    checkpoint_dir = os.path.join(checkpoint_dir, '{}'.format(args.load_epoch))
    variant = variant

    exp_runner = ExperimentRunner(variant)
    exp_runner.restore(checkpoint_dir)
    exp_runner.policy.set_always_deterministic()

    #environment = load_environment(variant)
    #policy = load_policy(checkpoint_dir, variant, environment)

    print("Policy loaded!")

    paths = rollouts(num_rollouts,
                     exp_runner.evaluation_environment,
                     exp_runner.policy,
                     path_length=max_path_length,
                     render_mode=render_mode,
                     squash=True)
    
    '''
    if video_save_path and render_mode == 'rgb_array':
        fps = 1 // getattr(environment, 'dt', 1/30)
        for i, path in enumerate(paths):
            video_save_dir = os.path.expanduser('/tmp/simulate_policy/')
            video_save_path = os.path.join(video_save_dir, f'episode_{i}.mp4')
            save_video(path['images'], video_save_path, fps=fps)
    '''

    return paths




if __name__ == '__main__':
    args = parse_args()

    if args.env == 'Hopper-v2':
        config = 'config.hopper'
    elif args.env == 'Swimmer-v2':
        config = 'config.swimmer'
    elif args.env == 'Walker2d-v2':
        config = 'config.walker2d'
    elif args.env == 'Ant-v2':
        config = 'config.ant'
    elif args.env == 'HalfCheetah-v2':
        config = 'config.halfcheetah'
    elif args.env == 'Humanoid-v2':
        config = 'config.humanoid'

    module = importlib.import_module(config)
    params = getattr(module, 'params')
    universe, domain, task = params['universe'], params['domain'], params['task']

    '''
    NUM_EPOCHS_PER_DOMAIN = {
        'Hopper': int(3e3),
        'HalfCheetah': int(3e3),
        'Walker2d': int(3e3),
        'Ant': int(3e3),
        'Humanoid': int(3e3),
        'Swimmer': int(3e3)
    }
    '''

    NUM_ITERS_PER_DOMAIN = {
        'Hopper': int(3),
        'HalfCheetah': int(3),
        'Walker2d': int(3),
        'Ant': int(3),
        'Humanoid': int(3),
        'Swimmer': int(3)
    }

    NUM_EPOCHS_PER_ITER = {
        'Hopper': int(1e3),
        'HalfCheetah': int(1e3),
        'Walker2d': int(1e3),
        'Ant': int(1e3),
        'Humanoid': int(1e3),
        'Swimmer': int(1e3)
    }

    params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_ITER[domain]
    params['kwargs']['n_initial_exploration_steps'] = 1000
    params['kwargs']['reparameterize'] = True
    params['kwargs']['lr'] = 3e-4
    params['kwargs']['target_update_interval'] = 1
    params['kwargs']['tau'] = 5e-3
    params['kwargs']['store_extra_policy_info'] = False
    params['kwargs']['action_prior'] = 'uniform'

    variant_spec = {
            'run_params': {
                'seed': args.seed,
                'checkpoint_at_end': True,
                'checkpoint_frequency': NUM_EPOCHS_PER_ITER[domain],
                'checkpoint_replay_pool': False,
            },
            'environment_params': {
                'training': {
                    'domain': domain,
                    'task': task,
                    'universe': universe,
                    'kwargs': {},
                },
                'evaluation': {
                    'domain': domain,
                    'task': task,
                    'universe': universe,
                    'kwargs': {},
                },
            },
            'policy_params': {
                'type': 'GaussianPolicy',
                'kwargs': {
                    'hidden_layer_sizes': (256, 256),
                    'squash': True,
                }
            },
            'Q_params': {
                'type': 'double_feedforward_Q_function',
                'kwargs': {
                    'hidden_layer_sizes': (256, 256),
                }
            },
            'algorithm_params': params,
            'replay_pool_params': {
                'type': 'SimpleReplayPool',
                'kwargs': {
                    'max_size': int(1e6),
                }
            },
            'sampler_params': {
                'type': 'SimpleSampler',
                'kwargs': {
                    'max_path_length': 1000,
                    'min_pool_size': 1000,
                    'batch_size': 256,
                    'squash': False,
                }
            },
        }
    
    set_seed(args.seed)
    path = simulate_policy(**vars(args), variant=variant_spec)
    rewards = [np.sum(_['rewards']) for _ in path]
    length = [len(_['rewards']) for _ in path]
    print("Env: {}, Example Traj: Reward-{}, Std-{}, Length-{}".format(args.env, np.mean(rewards), np.std(rewards), np.mean(length)))
    if args.traj_save_path is not None:
        save_path = os.path.join(args.traj_save_path, variant_spec['algorithm_params']['domain'])
        save_path = os.path.join(save_path, 'seed-{}'.format(args.seed))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 'exp_trajs_sac_{}.pkl'.format(args.num_rollouts))
        with open(save_path, 'wb') as f:
            pickle.dump(path, f)
    # print(path)
    
