import argparse
import joblib

from rllab.misc.instrument import VariantGenerator
from dgsl.algos.dgsl import DGSL
from dgsl.envs.multigoal import MultiGoalEnv
from dgsl.policies import GMMPolicy
from dgsl.replay_buffer.hindsight_latent_buffer import HindsightLatentBuffer
from dgsl.value_functions import NNQFunction, NNVFunction, NNDiscriminatorFunction
from dgsl.envs.env_utils import normalize_goal_env
from dgsl.misc.instrument import run_sac_experiment


SHARED_PARAMS = {
    "seed": [1],
    "lr": 1E-4,
    "discount": 0.99,
    "tau": 0.001,
    "K": 4,
    "n_latents": 32,
    "layer_size": 256,
    "batch_size": 256,
    "max_pool_size": 1E6,
    "n_sampled_goals": 1,
    "goal_strategy": 'final',
    "scale_reward": 3,
    "scale_entropy": 0.5,
    "n_train_repeat": 1,
    "epoch_length": 2500,
    "snapshot_mode": 'gap',
    "snapshot_gap": 5,
    "sync_pkl": True,
    'include_actions': False,
    'include_reward': True,
    'prefix': '2DNavigation',
    'exp_name': 'DGSL',
    'max_path_length': 30,
    'eval_n_episodes': 10,
    'n_epochs': 100,
    'goal_reward': 1.0,
    'distance_threshold': 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the embedding snapshot file.')
    parser.add_argument('--log_dir', type=str, default=None)
    arg_parser = parser.parse_args()

    return arg_parser


def get_variants():
    params = SHARED_PARAMS
    variant_generator = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            variant_generator.add(key, val)
        else:
            variant_generator.add(key, [val])

    return variant_generator


def run_experiment(variant):
    env = normalize_goal_env(
        MultiGoalEnv(
            goal_reward=variant['goal_reward'],
            distance_threshold=variant['distance_threshold']
        )
    )
    env_spec = env.goal_env_spec()

    goal_dim = env.observation_space.flat_dim_with_keys(
        keys=('desired_goal',)
    )       # goal dim(g)
    latent_dim = goal_dim

    data = joblib.load(args.file)
    embedding = data['embedding']
    print('embedding retrived ...')

    her_pool = HindsightLatentBuffer(
        env=env,
        max_replay_buffer_size=variant['max_pool_size'],
        n_sampled_goals=variant["n_sampled_goals"],
        goal_strategy=variant["goal_strategy"],
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=env_spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env_spec,
        hidden_layer_sizes=[M, M],
    )

    discriminator = NNDiscriminatorFunction(
        env_spec=env_spec,
        hidden_layer_sizes=[M, M],
        num_skills=latent_dim,
    )

    policy = GMMPolicy(
        env_spec=env_spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    base_kwargs = dict(
        min_pool_size=variant['epoch_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=variant['eval_n_episodes'],
        eval_deterministic=True,
    )

    algorithm = DGSL(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=her_pool,
        discriminator=discriminator,
        embedding=embedding,
        qf=qf,
        vf=vf,

        lr=variant['lr'],
        scale_reward=variant['scale_reward'],
        scale_entropy=variant['scale_entropy'],
        n_latents=variant['n_latents'],
        discount=variant['discount'],
        tau=variant['tau'],

        save_full_state=False,
        include_actions=variant['include_actions'],
        include_extrinsic_reward=variant['include_reward'],
    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode='local',
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + variant['exp_name'],
            exp_name=variant['prefix'] + '-' + variant['exp_name'] + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )


if __name__ == '__main__':
    args = parse_args()
    vg = get_variants()
    launch_experiments(vg)
