params = {
    'type': 'SAC',
    'universe': 'gym',
    'domain': 'HalfCheetah',
    'task': 'v2',

    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'target_entropy': -3,
    }
}
