import argparse

from rllab.misc.instrument import VariantGenerator
from dgsl.embeddings.gaussian_embedding import Embedding
from dgsl.envs.env_utils import normalize_goal_env
from dgsl.envs.multigoal import MultiGoalEnv
from dgsl.misc.instrument import run_sac_experiment


PARAMS = {
    "seed": [1, 2, 3, 4, 5],
    "layer_size": 100,
    "lr": 2e-5,
    "snapshot_mode": 'gap',
    "snapshot_gap": 5,
    "sync_pkl": True,
    'prefix': '2DNavigation',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_itrs', type=int, default=500)
    parser.add_argument('--log_dir', type=str, default=None)
    arg_parser = parser.parse_args()

    return arg_parser


def get_variants():
    params = PARAMS
    variant_generator = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            variant_generator.add(key, val)
        else:
            variant_generator.add(key, [val])

    return variant_generator


def run_experiment(variant):
    env = normalize_goal_env(MultiGoalEnv())
    goal_dim = len(env.sample_goal())
    M = variant['layer_size']

    embedding = Embedding(
        env=env,
        dim_g=goal_dim,
        lr=variant['lr'],
        hidden_layers_sizes=[2*M, 2*M, M]
    )

    embedding.train(n_itrs=args.n_itrs)


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode='local',
            variant=variant,
            exp_prefix="Embedding/" + variant['prefix'],
            exp_name=variant['prefix'] + '-Embedding-' + str(i).zfill(2),
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
