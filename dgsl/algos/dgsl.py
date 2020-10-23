"""Diverse goal-specific primitives Learning"""

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides

from dgsl.algos.sac import SAC
from dgsl.sampler.sampler_latent import rollouts

import gtimer as gt
import numpy as np
import tensorflow as tf


EPS = 1E-6


class DGSL(SAC):

    def __init__(self,
                 base_kwargs,
                 env,
                 policy,
                 discriminator,
                 embedding,
                 qf,
                 vf,
                 pool,
                 plotter=None,
                 lr=3E-3,
                 scale_reward=1,
                 scale_entropy=1,
                 n_latents=10,
                 discount=0.99,
                 tau=0.01,
                 save_full_state=False,
                 include_actions=False,
                 include_extrinsic_reward=True):
        """
        Args:
            base_kwargs (dict): dictionary of base arguments that are directly
                passed to the base `RLAlgorithm` constructor.
            env (`GoalEnv`): gym_env.GoalEnv object
            policy (`GMMPolicy`): A policy function approximator π(a|s,z).
            discriminator (`NNDiscriminatorFunction`): A discriminator q(z|s,g).
            embedding (`Embedding`): A trained embedding with given env goal space
            qf (`ValueFunction`): Q-function approximator Q(s,z,a).
            vf (`ValueFunction`): Soft value function approximator V(s,z).
            pool (`HindsightReplayBuffer`): Hindsight Replay buffer to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            scale_reward (`float`): Scaling factor for raw reward.
            scale_entropy (`float`): Scaling factor for entropy.
            n_latents (`int`): Number of latents sampled to train under a goal
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            save_full_state (`bool`): If True, save the full class in the
                snapshot. See `self.get_snapshot` for more information.
            include_actions (`bool`): Whether to pass actions to the discriminator.
            include_extrinsic_reward (`bool`): Whether the scaled reward will be
                included into surrogate reward
        """

        Serializable.quick_init(self, locals())
        super(SAC, self).__init__(**base_kwargs)

        self._env = env
        self._policy = policy
        self._discriminator = discriminator
        self._embedding = embedding
        self._qf = qf
        self._vf = vf
        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._discriminator_lr = lr
        self._qf_lr = lr
        self._vf_lr = lr
        self._scale_reward = scale_reward
        self._scale_entropy = scale_entropy
        self._n_latents = n_latents
        self._discount = discount
        self._tau = tau

        self._save_full_state = save_full_state
        self._include_actions = include_actions
        self._include_extrinsic_reward = include_extrinsic_reward

        self._Da = self._env.action_space.flat_dim
        self._Do = self._env.observation_space.flat_dim_with_keys(
            keys=('observation', 'desired_goal')
        )       # Do = dim(s) + dim(g)
        Dg = self._env.observation_space.flat_dim_with_keys(
            keys=('desired_goal',))
        self._Dg = Dg
        self._Dz = Dg

        self._training_ops = list()

        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_discriminator_update()
        self._init_target_ops()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
            - goal
        """

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )       # (s, z)

        self._obs_next_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='next_observation',
        )       # (s', z)

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._reward_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='rewards',
        )

        self._terminal_pl = tf.placeholder(
            tf.float32,
            shape=[None],
            name='terminals',
        )

        self._goal_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Dg],
            name='goal',
        )

    def _split_obs(self):
        return tf.split(self._obs_pl, [self._Do - self._Dz, self._Dz], 1)   # [N, Ds], [N, Dz]

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equation (10) in [1], for further information of the
        Q-function update rule.
        """

        self._qf_t = self._qf.get_output_for(
            self._obs_pl, self._action_pl, reuse=True)  # N

        (obs, latent) = self._split_obs()
        obs_and_goal = tf.concat([obs, self._goal_pl], axis=1)  # s||g, [N, Ds + Dg]

        # surrogate q(z|s,g)
        if self._include_actions:
            logits = self._discriminator.get_output_for(
                obs_and_goal, self._action_pl, reuse=True)
        else:
            logits = self._discriminator.get_output_for(
                obs_and_goal, reuse=True)       # N x Dz

        # compute surrogate reward
        surrogate_reward_pl = -1 * tf.reduce_sum((latent - logits)**2, axis=1)      # N
        surrogate_reward_pl = tf.check_numerics(surrogate_reward_pl,
                                                'Check numerics: reward_pl')
        if self._include_extrinsic_reward:
            surrogate_reward_pl += self._scale_reward * self._reward_pl
            # surrogate reward  # N

        self._surrogate_reward_pl = surrogate_reward_pl

        with tf.variable_scope('target'):
            vf_next_target_t = self._vf.get_output_for(self._obs_next_pl)  # N
            self._vf_target_params = self._vf.get_params_internal()

        ys = tf.stop_gradient(
            surrogate_reward_pl + (1 - self._terminal_pl) * self._discount * vf_next_target_t
        )  # N      # target Q-value

        self._td_loss_t = 0.5 * tf.reduce_mean((ys - self._qf_t)**2)

        qf_train_op = tf.train.AdamOptimizer(self._qf_lr).minimize(
            loss=self._td_loss_t,
            var_list=self._qf.get_params_internal()
        )

        self._training_ops.append(qf_train_op)

    def _init_actor_update(self):
        """Create minimization operations for policy and state value functions.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and value functions with gradient descent, and appends them to
        `self._training_ops` attribute.

        In principle, there is no need for a separate state value function
        approximator, since it could be evaluated using the Q-function and
        policy. However, in practice, the separate function approximator
        stabilizes training.

        See Equations (8, 13) in [1], for further information
        of the value function and policy function update rules.
        """

        self._policy_dist = self._policy.get_distribution_for(
            self._obs_pl, reuse=True)
        log_pi_t = self._policy_dist.log_p_t  # N

        self._vf_t = self._vf.get_output_for(self._obs_pl, reuse=True)  # N
        self._vf_params = self._vf.get_params_internal()

        log_target_t = self._qf.get_output_for(
            self._obs_pl, tf.tanh(self._policy_dist.x_t), reuse=True)  # N
        corr = self._squash_correction(self._policy_dist.x_t)
        corr = tf.check_numerics(corr, 'Check numerics: corr')

        scaled_log_pi = self._scale_entropy * (log_pi_t - corr)

        self._kl_surrogate_loss_t = tf.reduce_mean(log_pi_t * tf.stop_gradient(
            scaled_log_pi - log_target_t + self._vf_t)
        )

        self._vf_loss_t = 0.5 * tf.reduce_mean(
            (self._vf_t - tf.stop_gradient(log_target_t - scaled_log_pi))**2
        )

        policy_train_op = tf.train.AdamOptimizer(self._policy_lr).minimize(
            loss=self._kl_surrogate_loss_t + self._policy_dist.reg_loss_t,
            var_list=self._policy.get_params_internal()
        )

        vf_train_op = tf.train.AdamOptimizer(self._vf_lr).minimize(
            loss=self._vf_loss_t,
            var_list=self._vf_params
        )

        self._training_ops.append(policy_train_op)
        self._training_ops.append(vf_train_op)

    def _init_discriminator_update(self):
        (obs, latent) = self._split_obs()
        obs_and_goal = tf.concat([obs, self._goal_pl], axis=1)  # s||g, [N, Ds + Dg]

        # surrogate q(z|s,g)
        if self._include_actions:
            logits = self._discriminator.get_output_for(
                obs_and_goal, self._action_pl, reuse=True)
        else:
            logits = self._discriminator.get_output_for(
                obs_and_goal, reuse=True)       # [N, Dz]

        self._discriminator_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum((latent - logits)**2, axis=1)
        )

        optimizer = tf.train.AdamOptimizer(self._discriminator_lr)
        discriminator_train_op = optimizer.minimize(
            loss=self._discriminator_loss,
            var_list=self._discriminator.get_params_internal()
        )

        self._training_ops.append(discriminator_train_op)

    @overrides
    def _evaluate(self, epoch):
        """
        Perform evaluation for current policy.
        :param epoch: The epoch number.
        :return: None
        """

        if self._eval_n_episodes < 1:
            return

        goal = self._eval_env.sample_goal()
        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy, self._embedding, goal=goal,
                             path_length=self._max_path_length, n_paths=self._eval_n_episodes)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]
        success = [True in path['dones'] for path in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))
        logger.record_tabular('test-success-rate', np.mean(success))

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        batch = self._pool.random_batch(self._batch_size)
        self.log_diagnostics(batch)

    def _train(self, env, policy, pool):
        """When training our policy expects an augmented observation."""
        self._init_training(env, policy, pool)

        with self._sess.as_default():
            # reset with goal
            goal = env.sample_goal()
            observation = env.reset(goal=goal)
            policy.reset()

            # sample z ~ p(z|g)
            z = self._embedding.get_z(goal=goal)

            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0
            trajectory = []
            z_indx = 0

            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1), save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)
                path_length_list = []

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    # flatten observation with given latent z
                    aug_obs = np.concatenate((observation['observation'], z))
                    action, _ = policy.get_action(aug_obs)
                    next_ob, reward, terminal, info = env.step(action)

                    # assert all(next_ob['desired_goal'] == goal)
                    assert reward == env.compute_reward(next_ob['achieved_goal'],
                                                        next_ob['desired_goal'],
                                                        info)

                    path_length += 1
                    path_return += reward

                    trajectory.append(
                        (observation,
                         action,
                         reward,
                         next_ob,
                         terminal)
                    )

                    if terminal or path_length >= self._max_path_length:
                        path_length_list.append(path_length)

                        # add hindsight samples
                        self._pool.add_hindsight_episode(
                            episode=trajectory,
                            embedding=self._embedding,
                            latent=z,
                            goal=goal,
                        )

                        z_indx += 1
                        if z_indx >= self._n_latents:
                            goal = env.sample_goal()
                            z_indx = 0
                        z = self._embedding.get_z(goal=goal)

                        observation = env.reset(goal=goal)
                        policy.reset()

                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return
                        path_return = 0
                        n_episodes += 1
                        trajectory = []

                    else:
                        observation = next_ob

                    gt.stamp('sample')

                    if self._pool.size >= self._min_pool_size:
                        for i in range(self._n_train_repeat):
                            batch = self._pool.random_batch(self._batch_size)
                            self._do_training(iteration, batch)

                    gt.stamp('train')

                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('episodes', n_episodes)
                logger.record_tabular('steps', iteration)  # also record total steps
                logger.record_tabular('max-path-return', max_path_return)
                logger.record_tabular('last-path-return', last_path_return)
                logger.record_tabular('pool-size', self._pool.size)
                logger.record_tabular('path-length', np.mean(path_length_list))

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            env.terminate()

    def _get_feed_dict(self, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._obs_pl: batch['observations'],
            self._action_pl: batch['actions'],
            self._obs_next_pl: batch['next_observations'],
            self._reward_pl: batch['rewards'],
            self._terminal_pl: batch['terminals'],
            self._goal_pl: batch['goals'],
        }

        return feed_dict

    @overrides
    def log_diagnostics(self, batch):
        """Record diagnostic information to the logger.

        Records mean and standard deviation of Q-function and state
        value function, the TD-loss (mean squared Bellman error), and the
        discriminator loss (cross entropy) for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """
        feed_dict = self._get_feed_dict(batch)
        log_pairs = [
            ('qf', self._qf_t),
            ('vf', self._vf_t),
            ('bellman-error', self._td_loss_t),
            ('discriminator-loss', self._discriminator_loss),
            ('vf-loss', self._vf_loss_t),
            ('kl-surrogate-loss', self._kl_surrogate_loss_t),
            ('policy-reg-loss', self._policy_dist.reg_loss_t),
            ('surrogate_reward', self._surrogate_reward_pl),
        ]

        log_ops = [op for (name, op) in log_pairs]
        log_names = [name for (name, op) in log_pairs]
        log_vals = self._sess.run(log_ops, feed_dict)
        for (name, val) in zip(log_names, log_vals):
            if np.isscalar(val):
                logger.record_tabular(name, val)
            else:
                logger.record_tabular('%s-avg' % name, np.mean(val))
                logger.record_tabular('%s-min' % name, np.min(val))
                logger.record_tabular('%s-max' % name, np.max(val))
                logger.record_tabular('%s-std' % name, np.std(val))
        # logger.record_tabular('z-entropy', scipy.stats.entropy(self._p_z))

        self._policy.log_diagnostics(batch)
        if self._plotter:
            self._plotter.draw()

    @overrides
    def get_snapshot(self, epoch):
        """Return loggable snapshot of the SAC algorithm.

        If `self._save_full_state == True`, returns snapshot of the complete
        SAC instance. If `self._save_full_state == False`, returns snapshot
        of policy, Q-function, state value function, and environment instances.
        """

        if self._save_full_state:
            return dict(
                epoch=epoch,
                algo=self,
            )
        else:
            return dict(
                epoch=epoch,
                policy=self._policy,
                qf=self._qf,
                vf=self._vf,
                env=self._env,
                discriminator=self._discriminator,
            )

    def __getstate__(self):
        """Get Serializable state of the RLALgorithm instance."""

        d = Serializable.__getstate__(self)
        d.update({
            'qf-params': self._qf.get_param_values(),
            'vf-params': self._vf.get_param_values(),
            'discriminator-params': self._discriminator.get_param_values(),
            'policy-params': self._policy.get_param_values(),
            'pool': self._pool.__getstate__(),
            'env': self._env.__getstate__(),
        })
        return d

    def __setstate__(self, d):
        """Set Serializable state fo the RLAlgorithm instance."""

        Serializable.__setstate__(self, d)
        self._qf.set_param_values(d['qf-params'])
        self._vf.set_param_values(d['qf-params'])
        self._discriminator.set_param_values(d['discriminator-params'])
        self._policy.set_param_values(d['policy-params'])
        self._pool.__setstate__(d['pool'])
        self._env.__setstate__(d['env'])
