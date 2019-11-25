# CarRacing using PPO with Continuous Actions

### Introduction
This repository contains a simple version of PPO with Continuous actions implemented in Tensorflow 2 that can be used on Box 2d environments.
It follows the old PPO1 approach that uses open MPI to average gradients across GPUs. 
The reason it is created this way is so that with a little change it can be used as a layer in other models.

### Comparision with Stable Baselines

Here I compared my model with a Simple CNNPolicy using Stable Baselines. I used the VecFrameStack to stack 4 frames to be used as the input.
I did not use a LSTM or any other policy as I wanted to compare a basic model. We got similar results.

#### Custom Model Rewards
![Custom Model Rewards](Custom%20Model.png)

#### Stable Baselines Rewards

![Stable Baselines Rewards](Stable%20Baselines.png)



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
 
