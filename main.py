import argparse
import importlib
import runner
import os
from softlearning.misc.utils import set_seed
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint/')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--video_dir', type=str, default='./log/')
    args = parser.parse_args()

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
    elif args.env == 'InvertedPendulum-v2':
        config = 'config.invertedpendulum'
    elif args.env == 'InvertedDoublePendulum-v2':
        config = 'config.inverteddoublependulum'
    elif args.env == 'Pendulum':
        config = 'config.pendulum'

    module = importlib.import_module(config)
    params = getattr(module, 'params')
    universe, domain, task = params['universe'], params['domain'], params['task']

    log_dir = Path(args.log_dir).joinpath(domain)
    ckp_dir = Path(args.checkpoint_dir).joinpath(domain)
    video_dir = Path(args.video_dir).joinpath(domain)

    log_dir.mkdir(exist_ok=True, parents=True)
    ckp_dir.mkdir(exist_ok=True, parents=True)
    video_dir.mkdir(exist_ok=True, parents=True)

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
        'Swimmer': int(3),
        'InvertedPendulum': int(3),
        'InvertedDoublePendulum': int(3),
        'Pendulum': int(3),
    }

    NUM_EPOCHS_PER_ITER = {
        'Hopper': int(1e3),
        'HalfCheetah': int(1e3),
        'Walker2d': int(1e3),
        'Ant': int(1e3),
        'Humanoid': int(1e3),
        'Swimmer': int(1e3),
        'InvertedPendulum': int(1e2),
        'InvertedDoublePendulum': int(1e2),
        'Pendulum': int(1e2),
    }

    # params['kwargs']['n_epochs'] = NUM_EPOCHS_PER_DOMAIN[domain]
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
                    # 'kwargs': {'normalize':False},
                    'kwargs': {},
                },
                'evaluation': {
                    'domain': domain,
                    'task': task,
                    'universe': universe,
                    # 'kwargs': {'normalize':False},
                    'kwargs': {},
                },
            },
            'policy_params': {
                'type': 'GaussianPolicy',
                'kwargs': {
                    'hidden_layer_sizes': (256, 256),
                    # 'squash': False,
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
                    # 'squash': False,
                    'squash': True,
                }
            },
        }

    train_iters = NUM_ITERS_PER_DOMAIN[domain]
    checkpoint_dir = os.path.join(args.checkpoint_dir, variant_spec['algorithm_params']['domain'])
    checkpoint_dir = os.path.join(checkpoint_dir, 'seed-{}'.format(args.seed))

    exp_runner = runner.ExperimentRunner(variant_spec)
    set_seed(args.seed)

    for it in range(train_iters):
        print("Iterations {}".format(it))
        diagnostics = exp_runner.train()
        save_path = os.path.join(checkpoint_dir, str((it+1)*NUM_EPOCHS_PER_ITER[domain]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        exp_runner.save(save_path)

    save_path = os.path.join(checkpoint_dir, 'final')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_runner.save(save_path)
