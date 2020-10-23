import numpy as np
import copy

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

from .replay_buffer import ReplayBuffer


class HindsightLatentBuffer(ReplayBuffer, Serializable):
    def __init__(self, env, max_replay_buffer_size,
                 n_sampled_goals=2, goal_strategy='future'):
        env_spec = env.goal_env_spec()
        max_replay_buffer_size = int(max_replay_buffer_size)

        Serializable.quick_init(self, locals())
        super(HindsightLatentBuffer, self).__init__(env_spec)

        self._env = env
        self._env_spec = env_spec

        self._goal_dim = len(env.sample_goal())
        self._latent_dim = self._goal_dim
        self._observation_dim = env_spec.observation_space.flat_dim     # Ds + Dg
        self._action_dim = env_spec.action_space.flat_dim
        self._max_buffer_size = max_replay_buffer_size
        self._n_sampled_goals = n_sampled_goals
        self._goal_strategy = goal_strategy

        self._observations = np.zeros((max_replay_buffer_size,
                                       self._observation_dim))      # (s, z)
        self._next_obs = np.zeros((max_replay_buffer_size,
                                   self._observation_dim))      # (s', z)
        self._actions = np.zeros((max_replay_buffer_size, self._action_dim))
        self._rewards = np.zeros(max_replay_buffer_size)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros(max_replay_buffer_size, dtype='uint8')
        self._goals = np.zeros((max_replay_buffer_size, self._goal_dim))

        self._top = 0
        self._size = 0

    def _sample_strategy_goal(self, episode, start_idx, strategy='final'):
        """
        sample a hindsight goal using strategy
        :param episode: an episode whose observations are of type Dict
        :param start_idx: current time-step t
        :param strategy: goal sample strategy
        :return: a pseudo goal to be substituted for the actual desired goal using strategy
        """
        if strategy == 'future':
            transition_idx = np.random.choice(np.arange(start_idx + 1, len(episode)))
            transition = episode[transition_idx]
        elif strategy == 'final':
            transition = episode[-1]
        else:
            raise NotImplementedError

        goal = transition[0]['achieved_goal']
        # transition has structure (o,a,r,o2,d)

        return goal

    def add_hindsight_episode(self, episode, embedding, latent, goal):
        """
        Add an episode to hindsight replay buffer. We implement HER here.
        :param episode: [(o,a,r,o2,d),(),...()] where o is of type Dict, rollout by the skill with given latent.
        :param latent : (`np.array`) corresponding latent z according to which this episode is generated.
        :param goal : (`np.array`) corresponding goal g from which the latent z is sampled,
                                    and this skill is trying to reach it.
        :param embedding : (`HindsightLatentBuffer`) object, a trained embedding.
        :return: None

        For efficiency, we use strategy 'final' only, to circumvent embedding inference in each step
        """
        episode_goal = self._sample_strategy_goal(episode=episode, start_idx=None, strategy='final')
        replaced_z = embedding.get_mu(episode_goal)
        assert len(replaced_z) == self._goal_dim

        for t, transition in enumerate(episode):
            obs, action, reward, next_obs, done = transition

            # augmented_obs here is (s,z) rather than (s,g)
            aug_obs = np.concatenate((obs['observation'], latent))
            next_aug_obs = np.concatenate((next_obs['observation'], latent))
            # assert all(obs['desired_goal'] == goal)
            self.add_sample(aug_obs, action, reward, done, next_aug_obs, goal)

            obs, action, reward, next_obs, done = copy.deepcopy(transition)
            aug_obs = np.concatenate((obs['observation'], replaced_z))
            next_aug_obs = np.concatenate((next_obs['observation'], replaced_z))
            reward = self._env.compute_reward(next_obs['achieved_goal'],
                                              episode_goal, info=None)

            self.add_sample(aug_obs, action, reward, done, next_aug_obs, episode_goal)

        self.terminate_episode()

    @overrides
    def add_sample(self, observation, action, reward, terminal,
                   next_observation, goal, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._goals[self._top] = goal

        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            goals=self._goals[indices],
        )

    @property
    def size(self):
        return self._size

    def __getstate__(self):
        d = super(HindsightLatentBuffer, self).__getstate__()
        d.update(dict(
            o=self._observations.tobytes(),
            a=self._actions.tobytes(),
            r=self._rewards.tobytes(),
            t=self._terminals.tobytes(),
            no=self._next_obs.tobytes(),
            g=self._goals.tobytes(),
            top=self._top,
            size=self._size,
        ))
        return d

    def __setstate__(self, d):
        super(HindsightLatentBuffer, self).__setstate__(d)
        self._observations = np.fromstring(d['o']).reshape(
            self._max_buffer_size, -1
        )
        self._next_obs = np.fromstring(d['no']).reshape(
            self._max_buffer_size, -1
        )
        self._actions = np.fromstring(d['a']).reshape(self._max_buffer_size, -1)
        self._rewards = np.fromstring(d['r']).reshape(self._max_buffer_size)
        self._terminals = np.fromstring(d['t'], dtype=np.uint8)
        self._goals = np.fromstring(d['g']).reshape(self._max_buffer_size, -1)
        self._bottom = d['bottom']
        self._top = d['top']
        self._size = d['size']
