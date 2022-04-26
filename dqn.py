import os
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_wrappers import BatchedPytorchFrameStack, PytorchLazyFrames, make_atari_deepmind
from baselines_wrappers import Monitor, DummyVecEnv
import msgpack
from msgpack_numpy import patch as msgback_numpy_patch
from torch.utils.tensorboard import SummaryWriter
msgback_numpy_patch()

from configs import *
from model import Network
from utils import *

# Numpy warnings ignore
np.seterr(all="ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
wandb = init_wandb()

make_env = lambda: Monitor(make_atari_deepmind("ALE/Breakout-v5", scale_values=True), allow_early_resets=True)

vec_env = DummyVecEnv([make_env for  _ in range(NUM_ENVS)])

env = BatchedPytorchFrameStack(vec_env, k = 4)

replay_buffer = deque(maxlen=BUFFER_SIZE)
epinfos_buffer = deque([], maxlen=100)

episode_count = 0

# summary_writer = SummaryWriter(LOG_DIR)


online_net = Network(env, device=device).to(device)
target_net = Network(env, device=device).to(device)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

# Initialize replay buffer
obses = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

    new_obses, rews, dones, infos = env.step(actions)
    for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

    obses = new_obses

# Main Training Loop
obses = env.reset()


LOGGER.info(f"{colorstr('Optimizer:')} {optimizer}")
LOGGER.info(f"{colorstr('BATCH_SIZE:')} {BATCH_SIZE}")
BATCH_SIZE
LOGGER.info(f"{colorstr('EPSILON_DECAY:')} {EPSILON_DECAY}")
LOGGER.info(f"{colorstr('EPSILON_START:')} {EPSILON_START}    -    {colorstr('EPSILON_END:')} {EPSILON_END}")
LOGGER.info(f"{colorstr('TARGET_UPDATE_FREQ:')} {TARGET_UPDATE_FREQ}")
LOGGER.info(f"{colorstr('SAVE_PATH:')} {SAVE_PATH}")

LOGGER.info(colorstr('black', 'bold', '%20s' + '%15s' * 4) % 
                    ('Training:', 'gpu_mem', 'AvgRew', 'AvgEpLen', 'Episodes'))

with tqdm(range(N_STEPS), total=N_STEPS, 
          bar_format='{desc} {percentage:>7.0f}%|{bar:20}{r_bar}{bar:-10b}',
          unit='step') as pbar:
    for step in pbar:
        epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        
        rnd_sample = random.random()

        if isinstance(obses[0], PytorchLazyFrames):
            act_obses = np.stack([o.get_frames() for o in obses])
            actions = online_net.act(act_obses, epsilon)
        else:
            actions = online_net.act(obses, epsilon)

        new_obses, rews, dones, infos = env.step(actions)
        for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, done, new_obs)
            replay_buffer.append(transition)

            if done:
                epinfos_buffer.append(info['episode'])
                episode_count += 1

        obses = new_obses
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        loss = online_net.criterion(transitions, target_net)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Target Net
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())

        rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g} GB'
        desc = ('%35s' + '%15.6g' * 3) % (mem, rew_mean, len_mean, episode_count)

        pbar.set_description_str(desc)
        wandb.log({
            "Average Reward": rew_mean,
            "Average Episode Length": rew_mean,
            "Episodes": episode_count
        })

        if step % SAVE_INTERVAL == 0 and step != 0:
            LOGGER.info(f"\n{colorstr('black', 'bold', f'Saving model at {step}...')}")
            online_net.save(os.path.join(SAVE_PATH, f"breakout_b32_at{step // SAVE_INTERVAL}k.pack"))