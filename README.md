# Diverse Goal-specific Skills Learning (DGSL)

Codes accompanying the paper: "Learning Diverse Goal-specific Transferable Skills in Latent Space".  
(The manuscript is still on-going)

This framework provides an implementation of DGSL algorithm for training diverse goal-specific transferable skills in latent space.

## Getting Started

### Prerequisites

1. To get everything installed correctly, you will first need to clone [rllab](https://github.com/rll/rllab), and have its path added to your PYTHONPATH environment variable.

```
cd <installation_path_of_rllab>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

2. Download [mujoco](https://www.roboti.us/index.html) (mjpro150 linux) and copy several files to rllab path: 

```
mkdir <installation_path_of_rllab>/rllab/vendor/mujoco
cp <installation_path_of_mujoco>/mjpro150/bin/libmujoco150.so <installation_path_of_rllab>/rllab/vendor/mujoco
cp <installation_path_of_mujoco>/mjpro150/bin/libglfw.so.3 <installation_path_of_rllab>/rllab/vendor/mujoco
```

3. Copy your mujoco license key (mjkey.txt) to rllab path:

```
cp <mujoco_key_folder>/mjkey.txt <installation_path_of_rllab>/rllab/vendor/mujoco
```

### Requirements

To install requirements, run:

```
pip install -r requirements.txt
```

and everything should be good to go.

## Toy Examples

### Training Agents

To train diverse goal-specific latent-conditioned policies in 2D-Navigation environment, run the following commands.

1. Train the latent embedding with the given environment:

```
python examples/embedding.py --n_itrs=1000
```

- `n_itrs` specifies the maximum number of training iterations. By default it's 500 if you remove the flag.
- The log(.csv) and model(.pkl) will be saved to the `data/` directory by default. But the output directory can also be specified with `--log_dir=<log-directory>`.

2. Train the latent-conditioned policy using the pre-trained latent embedding:

```
python examples/run_multigoal_dgsl.py <embedding-file-directory>
```

- `<embedding-file-directory>` specifies the directory of `.pkl` file that contains the pre-trained latent embedding.
- The log(.csv) and model(.pkl) will be saved to the `data/` directory by default. But the output directory can also be specified with `--log_dir=<log-directory>`.


### Visualizing Agents

To simulate the trained latent-conditioned policy, run:

```
python examples/visualize.py <model-file-directory> <embedding-file-directory> --max_path_length=20 --save_image
```

- `<model-file-directory>` specifies the directory of `.pkl` file that contains the trained latent-conditioned policy.
- `<embedding-file-directory>` specifies the directory of `.pkl` file that contains the trained latent embedding.
- `--max_path_length` specifies the maximum environment steps in simulation. By default it's 15 if you remove the flag.
- `--save_image` enables saving rendering images to `../viz_data` directory. If you remove the flag, it will only render the environment without saving rendering images.
