params = {
    'type': 'SAC',
    'universe': 'gym',
    'domain': 'Pendulum',
    'task': 'v0',

    'kwargs': {
        'epoch_length': 200,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'target_entropy': -1,
    }
}

