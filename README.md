# CarRacing using PPO with Continuous Actions

![Custom Model Rewards](Custom%20Model.png)

![Stable Baselines Rewards](Stable%20Baselines.png)

### Introduction
This repository contains a simple version of PPO with Continuous actions that can be used on Box 2d environments.
It follows the old PPO1 approach that uses open MPI to average gradients across GPUs. 
The reason it is created this way is so that with a little change it can be used as a layer in other models.

### Requirements
- gym
- gym[box2d]
- tensorflow-gpu 2.0.0
- horovod https://github.com/horovod/horovod/blob/master/docs/gpus.rst
- tensorflow_probability
- cv2

### Steps to run
- horovodrun -np 4 -H localhost:4 python train_ppo.py

This command will use 4 gpus and create one model on each. It will then average the gradients across GPUs. 
 
