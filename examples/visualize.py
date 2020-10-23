import argparse
import joblib
import tensorflow as tf
import numpy as np

from dgsl.sampler.sampler_latent import rollouts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str, help='Path to the snapshot file.')
    parser.add_argument('embedding_file', type=str, help='Path to the embedding snapshot file.')
    parser.add_argument('--max_path_length', type=int, default=15)
    parser.add_argument('--n_paths', type=int, default=10)
    parser.add_argument('--save_image', action='store_true', default=False)
    parser.add_argument('--deterministic', '-d', dest='deterministic', action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic', action='store_false')
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.model_file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        data_emb = joblib.load(args.embedding_file)
        embedding = data_emb['embedding']

        with policy.deterministic(args.deterministic):
            goals = np.array([[5, 0], [-5, 0], [0, 5], [0, -5]],
                             dtype=np.float32)
            for i in range(10):     # render 10 images
                paths = []
                for goal in goals:
                    ps = rollouts(env, policy, embedding, goal,
                                  args.max_path_length, n_paths=args.n_paths)
                    paths += ps
                env.render(paths, save_fig=args.save_image, name_idx=i+1)
