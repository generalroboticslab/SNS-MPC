# Learning Legged MPC with Smooth Neural Surrogates
<a href="https://samavmoore.github.io/">Sam Moore</a>, <a href="https://easoplee.github.io/">Easop Lee</a>, and <a href="http://boyuanchen.com/">Boyuan Chen</a> <br>
_Duke University_ <br>

<span style="font-size:17px; display: block; text-align: left;">
    <a href=https://generalroboticslab.com/SNS-MPC target="_blank" style="text-decoration: underline;">[Project Page]</a> 
    <a href=https://youtu.be/ViwE7hVG-J4?si=Nq25zR65D3Kooub8 target="_blank" style="text-decoration: underline;">[Video]</a>
    <a href=https://arxiv.org/abs/2601.12169 target="_blank" style="text-decoration: underline;">[arXiv]</a> <br>
</span>

### JAX: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18mjmXNuOK5XXlvSIaKOzolU1TXHmPw9q?usp=sharing})

### PyTorch: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16eYXHwyoadNLCVKasIjB0Qc5qTfh39as?usp=sharing})

<p align="center">
    <img src="media/website_fig_1.png" width="800"> <br>
</p>

### Overview
Learning and model predictive control can be complementary for legged robotics, but integrating learned dynamics with online planning remains difficult. Neural network dynamics suffer from three issues: stiff transitions are inherited from the training data, non-physical local nonsmoothness, and non-Gaussian dynamics model errors from rapid state changes. We address the first two with the smooth neural surrogate, a neural network with tunable smoothness that provides informative predictions and well-conditioned derivatives for trajectory optimization through contact. To address the third, we train these models using a heavy-tailed likelihood that we show to be a better match for empirical error distributions. Together, these choices substantially improve learned model predictive control. Across zero-shot locomotion tasks of increasing difficulty, smooth neural surrogates reduce cumulative cost on well-conditioned behaviors (typically 10–50%) and yield substantially larger gains in regimes where standard neural dynamics fail. In these regimes, smoothing enables reliable execution (from 0/5 to 5/5 success) and 2–50 lower cumulative cost, reflecting vast improvements in controller generalization.

<p align="center">
    <img src="media/website_fig_2.png" width="800"> <br>
</p>

## Content

Code release for the paper *Learning Legged MPC with Smooth Neural Surrogates*.

This public release is scoped to the simulation workflows used to train, evaluate, and deploy smooth neural surrogate models for Go2 locomotion in MuJoCo. Real-robot deployment utilities, paper plotting scripts, and multi-GPU experiment sweep launchers are intentionally excluded from the supported surface.

## Setup

Create the environment and install the package in editable mode:

```bash
conda env create -f environment.yaml
conda activate nnmpc
python -m pip install -e .
```

If your system has trouble with MuJoCo OpenGL rendering, the environment file already targets EGL. On some Ubuntu setups you may still need system OpenGL packages installed.


## Data

The default training config is set up to build its initial replay buffer using random actions rather than requiring a precomputed `.dill` file. However, this can be time consuming. When you run training for the first time, this initial buffer will be saved to the path you provide in the config:

```yaml
buffer_path: ./path_to_your_initial_buffer.dill
buffer_save: True
```

After this has been saved for reuse, the training script will load from it if it exists. If you want to start with a precomputed dataset, you need to set the following flag in the training config:

```yaml
buffer_path: ./path_to_your_initial_buffer.dill
buffer_save: False # will not regenerate the buffer if this is set to False, and will load from the provided path if it exists
```

## Parallelism

The current release uses CPU-based parallelism in MuJoCo for data collection, and GPU-based parallelism for training in JAX. In our training setup, we used a single GPU and a large number of CPU workers on a server with 128 CPU cores. Because of this, data collection may be the bottleneck for training on a local machine. In future releases, we may migrate to MJWarp/MJLab. Deployment and evaluation scripts are easy to run on a local machine.

## Supported Entry Points

Training:

```bash
python run_training.py --config ./Configs/Go2_dog/train.yaml --gpu 0
```

Simulation evaluation with MPPI and DIAL-MPC:

```bash
python run_evaluation.py --config ./Configs/Go2_dog/evaluate_mppi.yaml --gpu 0
```

Simulation evaluation with our GGN-MPC (spline shooter):

```bash
python run_evaluation.py --config ./Configs/Go2_dog/evaluate_spline_shooter.yaml --gpu 0
```

Simulation deployment with MPPI and DIAL-MPC:

```bash
python run_deployment_sim.py --config ./Configs/Go2_dog/deploy_lipschitz_mppi.yaml --gpu 0
```

Simulation deployment with our GGN-MPC (spline shooter):

```bash
python run_deployment_sim.py --config ./Configs/Go2_dog/deploy_lipschitz_spline_shooter.yaml --gpu 0
```

All supported scripts keep the existing `--gpu` flag and write logs under `Logs/`.

## Runtime Assets

The repository keeps the runtime assets needed by the supported workflows:

- `Mj_models/` contains the MuJoCo Go2 assets.
- `actuator_logs/actuator_network_params.npz` and `actuator_logs/actuator_network_200hz_params.npz` are the shipped actuator networks used by the simulation stack.
- `Trained_models/Primary/Ckpt/Lipschitz_2025-07-14_15-48-44/` contains a retained example checkpoint and the associated state/action/observation normalization artifacts used by the public configs.

## Actuator Model Provenance

`train_actuator_net.py` is retained as a secondary utility for reproducing actuator-model assets.

The repo keeps the actuator training artifacts it depends on:

- `actuator_logs/data_200hz.npz`
- `actuator_logs/log.pkl`

So, if you want to retrain the actuator model, you can run:

```bash
python train_actuator_net.py
```


## BibTeX

If you find this repo useful, please consider citing,
```
@misc{moore2026learningleggedmpc,
      title={Learning Legged MPC with Smooth Neural Surrogates}, 
      author={Samuel A. Moore and Easop Lee and Boyuan Chen},
      year={2026},
      eprint={2601.12169},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.12169}, 
}
```

## Acknowledgements
`This work is supported by the National Science Foundation Graduate Research Fellowship, by ARO under award W911NF2410405, and by DARPA TIAMAT program under award HR00112490419.`.