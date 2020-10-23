import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.envs.base import Env
from rllab.envs.env_spec import EnvSpec

from dgsl.envs.env_utils import Dict


class MultiGoalEnv(Env, Serializable):
    """
    Move a 2D point mass to one of the goal positions. Cost is the distance to
    the closest goal.

    State: position.
    Action: velocity.
    """
    def __init__(self, distance_threshold=1.0, goal_reward=1.0, init_sigma=0.1):
        super(MultiGoalEnv, self).__init__()
        Serializable.quick_init(self, locals())

        self.dynamics = PointDynamics(dim=2, sigma=0)
        self.init_mu = np.array((0, 0), dtype=np.float32)
        self.init_sigma = init_sigma
        self.goal_positions = np.array(
            [
                [5, 0],
                [-5, 0],
                [0, 5],
                [0, -5]
            ],
            dtype=np.float32
        )

        self.goal_reward = goal_reward
        self.distance_threshold = distance_threshold
        self.xlim = (-7, 7)
        self.ylim = (-7, 7)
        self.vel_bound = 1.

        self.reset()
        self.observation = None
        self.goal = None

        self._ax = None
        self._env_lines = list()
        self.fixed_plots = None
        self.dynamic_plots = []

    @overrides
    @property
    def observation_space(self):
        obs_space = dict(
            observation=self._get_2D_Box_space(),
            achieved_goal=self._get_2D_Box_space(),
            desired_goal=self._get_2D_Box_space(),
        )

        return Dict(obs_space)

    @overrides
    @property
    def action_space(self):
        return Box(
            low=-self.vel_bound,
            high=self.vel_bound,
            shape=(self.dynamics.a_dim,)
        )

    def get_current_obs(self, obs):
        return dict(
            observation=np.copy(obs),
            achieved_goal=np.copy(obs),
            desired_goal=np.copy(self.goal),
        )

    @overrides
    def reset(self, goal=None):
        """
        reset with given goal.
        if the goal is NOT given, sample a goal in the goal space.
        """
        unclipped_observation = self.init_mu + self.init_sigma * \
            np.random.normal(size=self.dynamics.s_dim)
        o_lb, o_ub = self.observation_space.spaces['observation'].bounds
        init_obs = np.clip(unclipped_observation, o_lb, o_ub)

        # reset with given goal
        if goal is None:
            self.goal = self.sample_goal()
        else:
            self.goal = np.copy(goal)

        observation = self.get_current_obs(init_obs)
        self.observation = copy.deepcopy(observation)

        return self.observation

    @overrides
    def step(self, action):
        action = action.ravel()

        a_lb, a_ub = self.action_space.bounds
        action = np.clip(action, a_lb, a_ub).ravel()

        next_obs = self.dynamics.forward(self.observation['observation'], action)
        o_lb, o_ub = self.observation_space.spaces['observation'].bounds
        next_obs = np.clip(next_obs, o_lb, o_ub)
        next_obs = self.get_current_obs(next_obs)
        self.observation = copy.deepcopy(next_obs)

        reward = self.compute_reward(
            self.observation['achieved_goal'],
            self.goal, None
        )

        # d = self.goal_distance(self.observation['achieved_goal'], self.goal)
        # done = d <= self.distance_threshold
        done = False

        info = {'pos': next_obs['observation']}

        return next_obs, reward, done, info

    def _init_plot(self):
        fig_env = plt.figure(figsize=(7, 7))
        self._ax = fig_env.add_subplot(111)
        self._ax.axis('equal')

        self._env_lines = []
        self._ax.set_xlim(-7, 7)
        self._ax.set_ylim(-7, 7)

        self._ax.set_title('Multigoal Environment')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')

        self._plot_position_cost(self._ax)

    @overrides
    def render(self, paths, save_fig=False, name_idx=0):
        """
        Render or save paths returned by rollouts.
        :param paths: (`list`) paths returned by sampler_latent.rollouts
        :param save_fig: (`bool`) Save render image or not.
                                  If True, the image will NOT be rendered.
        :param name_idx: (`int`) Image name index number. (if save_fig == True)
        :return:
        """
        if self._ax is None:
            self._init_plot()

        # noinspection PyArgumentList
        [line.remove() for line in self._env_lines]
        self._env_lines = list()

        for path in paths:
            positions = path["env_infos"]["pos"]
            xx = positions[:, 0]
            yy = positions[:, 1]
            self._env_lines += self._ax.plot(xx, yy, 'b')

        if save_fig:
            viz_data_path = '../viz_data/'
            if not os.path.exists(viz_data_path):
                os.mkdir(viz_data_path)

            fig = plt.gcf()
            fig.savefig(viz_data_path + 'multigoal-dgsl-' + str(name_idx) + '.png', bbox_inches='tight')
        else:
            plt.draw()
            plt.pause(0.01)

    def sample_goal(self):
        ind = np.random.randint(len(self.goal_positions))
        goal = self.goal_positions[ind]
        return np.copy(goal)

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = self.goal_distance(achieved_goal, goal)
        reward = -(d > self.distance_threshold).astype(np.float32)
        if d <= self.distance_threshold:
            reward += self.goal_reward

        return reward

    @staticmethod
    def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def _plot_position_cost(self, ax):
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self.xlim))
        y_min, y_max = tuple(1.1 * np.array(self.ylim))
        X, Y = np.meshgrid(
            np.arange(x_min, x_max, delta),
            np.arange(y_min, y_max, delta)
        )
        goal_costs = np.amin([
            (X - goal_x) ** 2 + (Y - goal_y) ** 2
            for goal_x, goal_y in self.goal_positions
        ], axis=0)
        costs = goal_costs

        contours = ax.contour(X, Y, costs, 20)
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        goal = ax.plot(self.goal_positions[:, 0],
                       self.goal_positions[:, 1], 'ro')
        return [contours, goal]

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def goal_env_spec(self, dict_keys=('observation', 'desired_goal')):
        """
        convert selected keys of a dict observation space to a Box space
        and return the corresponding env_spec.
        :param dict_keys: (`tuple`) desired keys that you would like to use.
        :return: (`EnvSpec`) converted object
        """
        assert isinstance(self.observation_space, Dict)
        obs_dim = np.sum([self.observation_space.spaces[key].flat_dim for key in dict_keys])
        obs_space = Box(-np.inf, np.inf, shape=(obs_dim,))

        return EnvSpec(
            observation_space=obs_space,
            action_space=self.action_space,
        )

    @overrides
    def log_diagnostics(self, paths):
        pass

    @overrides
    def horizon(self):
        return None

    def _get_2D_Box_space(self):
        return Box(
            low=np.array((self.xlim[0], self.ylim[0])),
            high=np.array((self.xlim[1], self.ylim[1])),
            shape=None,
        )


class PointDynamics(object):
    """
    State: position.
    Action: velocity.
    """
    def __init__(self, dim, sigma):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            np.random.normal(size=self.s_dim)
        return state_next
