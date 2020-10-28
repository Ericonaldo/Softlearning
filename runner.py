import tensorflow as tf
from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy_from_variant, get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.misc.utils import set_seed, initialize_tf_variables

import time, os, pickle
import tree, sys, glob, sys, copy, json

class ExperimentRunner:
    def __init__(self, variant):
        # set_seed(variant['run_params']['seed'])
        # self.experiment_id = variant['algorithm_params']['exp_name']
        # self.local_dir = os.path.join(variant['algorithm_params']['log_dir'], variant['algorithm_params']['domain'])

        self.variant = variant
        self.tf_session = session = tf.Session()
        tf.keras.backend.set_session(session)
        self._session = tf.keras.backend.get_session()
        
        self._built = False

    def build(self):
        environment_params = self.variant['environment_params']
        training_environment = self.training_environment = (get_environment_from_params(environment_params['training']))
        evaluation_environment = self.evaluation_environment = (
            get_environment_from_params(environment_params['evaluation'])
            if 'evaluation' in environment_params
            else training_environment)

        replay_pool = self.replay_pool = (get_replay_pool_from_variant(self.variant, training_environment))
        sampler = self.sampler = get_sampler_from_variant(self.variant)
        Qs = self.Qs = get_Q_function_from_variant(self.variant, training_environment)
        policy = self.policy = get_policy_from_variant(self.variant, training_environment, Qs)
        initial_exploration_policy = self.initial_exploration_policy = (get_policy('UniformPolicy', training_environment))

        #### get termination function
        # domain = environment_params['training']['domain']
        # static_fns = static[domain.lower()]
        ####

        self.algorithm = get_algorithm_from_variant(
            variant=self.variant,
            training_environment=training_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=initial_exploration_policy,
            Qs=Qs,
            pool=replay_pool,
            # static_fns=static_fns,
            sampler=sampler,
            session=self._session,
            log_file='./log/%s/%d.log' % (self.variant['algorithm_params']['domain'], time.time())
        )

        initialize_tf_variables(self._session, only_uninitialized=True)

        self._built = True

    def train(self):
        if not self._built:
            self.build()
        
        diagnostics = self.algorithm.train()

        return diagnostics

    @staticmethod
    def _pickle_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    @staticmethod
    def _algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'algorithm')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy')
    @staticmethod
    def _value_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'value')

    def _save_replay_pool(self, checkpoint_dir):
        if not self.variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        replay_pool_save_path = self._replay_pool_save_path(checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_save_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        if not self.variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_save_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _save_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'wb') as f:
            pickle.dump(self.sampler, f)

    def _restore_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'rb') as f:
            sampler = pickle.load(f)

        self.sampler.__setstate__(sampler.__getstate__())
        self.sampler.initialize(
            self.training_environment, self.policy, self.replay_pool)

    def _save_value_functions(self, checkpoint_dir):
        save_path = self._value_save_path(checkpoint_dir)
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    save_path, '-'.join(('Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.Qs)

    def _restore_value_functions(self, checkpoint_dir):
        save_path = self._value_save_path(checkpoint_dir)
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    save_path, '-'.join(('Q', *[str(x) for x in path])))),
            self.Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        self.policy.save(save_path+"/checkpoint")

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        
        latest = tf.train.latest_checkpoint(save_path)
        status = self.policy.load(latest)
        #status.assert_consumed().run_restore_ops()

    def _save_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint", session=self.tf_session)

        state = self.algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.algorithm._Q_optimizers),
            tuple(self.Qs),
        )

        self.algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.algorithm._log_alpha), self.algorithm._log_alpha
        )])
        self.algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        
        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        # status.assert_consumed().run_restore_ops()

    def save(self, checkpoint_dir):
        """Implements the checkpoint save logic."""
        self._save_replay_pool(checkpoint_dir)
        self._save_sampler(checkpoint_dir)
        self._save_value_functions(checkpoint_dir)
        self._save_policy(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def restore(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        self.build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_algorithm(checkpoint_dir)

        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True
