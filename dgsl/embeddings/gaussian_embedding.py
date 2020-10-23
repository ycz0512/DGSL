""" Gaussian latent embedding. """

import tensorflow as tf
import numpy as np

from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized

from dgsl.misc.mlp import mlp
from dgsl.misc import tf_utils


LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -5


class Embedding(Parameterized, Serializable):

    def __init__(self, env, dim_g, lr=1E-3, hidden_layers_sizes=(100, 100)):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

        assert hasattr(env, 'sample_goal')
        self._env = env

        self._dim_g = dim_g
        self._dim_z = dim_g
        self._layer_sizes = list(hidden_layers_sizes) + [2 * self._dim_z]
        self._name = 'embedding'
        self._lr = lr

        self._training_ops = list()
        self._create_placeholders()

        self._zs, self._log_p_zs, self._mus = \
            self.get_output_for(self._goals_pl, reuse=False)
        # M x K x Dz, M x K x 1, M x K x Dz
        self._fetches = [tf.exp(self._log_p_zs), self._zs, self._mus]

        self._create_embedding_update()

        # Initialize all uninitialized variables. This prevents initializing
        # pre-trained policy and qf and vf variables.
        self._sess = tf_utils.get_default_session()
        uninit_vars = []
        for var in tf.global_variables():
            try:
                self._sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninit_vars.append(var)
        self._sess.run(tf.variables_initializer(uninit_vars))

    def _create_placeholders(self):
        self._goals_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._dim_g],
            name='goal',
        )

        self._zs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._dim_z],
            name='latent',
        )

    def _create_embedding_update(self):
        p_z = tf.reduce_mean(tf.exp(self._log_p_zs), axis=0)   # K x 1
        self._entropy_z = -1 * tf.reduce_sum(p_z * tf.log(p_z))
        emb_loss = -1 * self._entropy_z

        emb_train_op = tf.train.AdamOptimizer(self._lr).minimize(
            loss=emb_loss,
            var_list=self.get_params_internal(),
        )

        self._training_ops.append(emb_train_op)

    def _get_feed_dict(self, env, batch_size):
        zs = list()
        gs = [env.sample_goal() for _ in range(batch_size)]
        for g in gs:
            zs.append(self.get_z(goal=g))

        feed_dict = {self._goals_pl: gs, self._zs_pl: zs}
        return feed_dict

    def _do_training(self, batch_size):
        feed_dict = self._get_feed_dict(
            env=self._env, batch_size=batch_size)

        self._sess.run(self._training_ops, feed_dict)
        latent_entropy = self._sess.run(
            self._entropy_z, feed_dict=feed_dict)

        return latent_entropy

    @staticmethod
    def _create_log_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * tf.exp(-log_sig_t)  # ... x D
        quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1)
        # ... x (None)

        log_z = tf.reduce_sum(log_sig_t, axis=-1)  # ... x (None)
        D_t = tf.cast(tf.shape(mu_t)[-1], tf.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (None)

    def get_output_for(self, *inputs, reuse=False):
        """
        construct computational graph w.r.t. log p(z|g) and z ~ log p(z|g)
        :param inputs: input placeholders
        :param reuse: reuse variables under scope=self._name or not
        :return: given g, sample a latent z ~ log p(z|g)
                 given g and z, compute log probability log p(z|g)
        """
        with tf.variable_scope(self._name, reuse=reuse):
            mu_and_log_sig_t = mlp(
                inputs=inputs,
                layer_sizes=self._layer_sizes,
                output_nonlinearity=None,
            )  # M x 2*dim_z

            M = tf.shape(mu_and_log_sig_t)[0]
            K = tf.shape(self._zs_pl)[0]

            mu_and_log_sig_t = tf.expand_dims(
                mu_and_log_sig_t, axis=1)  # M x 1 x 2*dim_z
            mu_and_log_sig_t = tf.tile(
                mu_and_log_sig_t, multiples=[1, K, 1])     # M x K x 2*dim_z

            mu_t = mu_and_log_sig_t[..., 0:self._dim_z]  # M x K x dim_z
            log_sig_t = mu_and_log_sig_t[..., self._dim_z:]  # M x K x dim_z

            log_sig_t = tf.minimum(log_sig_t, LOG_SIG_CAP_MAX)
            log_sig_t = tf.maximum(log_sig_t, LOG_SIG_CAP_MIN)

            # sample a latent z ~ p(z|g)
            sig_t = tf.exp(log_sig_t)
            zs = tf.stop_gradient(
                mu_t + sig_t * tf.random_normal(shape=(M, K, self._dim_z))
            )  # M x K x dim_z

            # compute log p(z|g)
            zs_pl = tf.expand_dims(self._zs_pl, axis=0)     # 1 x K x dim_z
            zs_pl = tf.tile(zs_pl, multiples=[M, 1, 1])     # M x K x dim_z
            log_p_zs = self._create_log_gaussian(mu_t, log_sig_t, zs_pl)  # M x K x 1

        return zs, log_p_zs, mu_t

    def train(self, n_itrs=1000, batch_size=128):
        with self._sess.as_default():
            for itr in range(n_itrs + 1):
                logger.push_prefix('iteration #%d | ' % itr)
                latent_entropy = self._do_training(batch_size=batch_size)

                params = self.get_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.record_tabular('iteration', itr)
                logger.record_tabular('latent_entropy', latent_entropy)
                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

    def get_outputs(self, goals, zs):
        feeds = {self._goals_pl: goals,    # M x dim_g
                 self._zs_pl: zs}      # K x dim_z
        p_zs, zs, mus = tf.get_default_session().run(self._fetches, feeds)
        # p(z|g) [M, K, 1], z~p(z|g) [M, K, Dz], mu(g) [M, K, Dz]

        return p_zs, zs, mus

    def get_p_z(self, goal, z):
        """ given g and z, compute p(z|g)."""
        p_zs, _, _ = self.get_outputs(goal[None], z[None])
        return p_zs[0][0]

    def get_z(self, goal):
        """ given g, sample a latent z ~ p(z|g)."""
        zs_feed = np.zeros(shape=(1, self._dim_z))
        _, zs, _ = self.get_outputs(goal[None], zs_feed)
        return zs[0][0]

    def get_mu(self, goal):
        """ given g, compute mu(g)."""
        mus_feed = np.zeros(shape=(1, self._dim_z))
        _, _, mus = self.get_outputs(goal[None], mus_feed)
        return mus[0][0]

    def get_params_internal(self, **tags):
        """ Return embedding parameters."""
        if len(tags) > 0:
            raise NotImplementedError

        scope = tf.get_variable_scope().name
        scope += '/' + self._name + '/' if len(scope) else self._name + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )

    def get_snapshot(self, itr):
        """ Return loggable snapshot of the embedding."""
        return dict(itr=itr, embedding=self)
