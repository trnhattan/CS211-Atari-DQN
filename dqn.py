import argparse
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
# from torch.utils.tensorboard import SummaryWriter
msgback_numpy_patch()

from configs import *
from model import Network
from utils import *

# Numpy warnings ignore
np.seterr(all="ignore")

def train(env_id="BreakoutNoFrameskip-v4", 
          model = None,
          resume=False, 
          file_weight_path=None,
          file_saveName="breakout_b32_at", 
          project="Deep-Q-Learning-Network", 
          entity='devzxje', 
          name="DQN-v1", 
          id_name="DQN-v1",
          session_resume=None,
          relogin=False,
          device="cpu"):

    wandb = init_wandb(project=project, 
                       entity=entity, 
                       name=name, 
                       id=id_name,
                       session_resume=session_resume,
                       relogin=relogin)

    make_env = lambda: Monitor(make_atari_deepmind(env_id=env_id, scale_values=True), allow_early_resets=True)

    vec_env = DummyVecEnv([make_env for  _ in range(NUM_ENVS)])

    env = BatchedPytorchFrameStack(vec_env, k = 4)

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epinfos_buffer = deque([], maxlen=100)

    episode_count = 0

    # summary_writer = SummaryWriter(LOG_DIR)

    online_net = Network(env, device=device, model=model).to(device)
    target_net = Network(env, device=device, model=model).to(device)

    if resume and file_weight_path:
        LOGGER.info(colorstr('black', 'bold', f"Loading weights from {file_weight_path}..."))
        online_net.load(file_weight_path)

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


    if model == 'base':
        LOGGER.info(f"{colorstr('Model: Normal Deep Q-Network')}")
    elif model == 'double':
        LOGGER.info(f"{colorstr('Model: Double Deep Q-Network')}")
    elif model == 'dueling':
        LOGGER.info(f"{colorstr('Model: Dueling Deep Q-Network')}")
    else:
        raise NotImplementedError(f"{model} not found")


    LOGGER.info(f"{colorstr('Optimizer:')} {optimizer}")
    LOGGER.info(f"{colorstr('BATCH_SIZE:')} {BATCH_SIZE}")
    BATCH_SIZE
    LOGGER.info(f"{colorstr('EPSILON_DECAY:')} {EPSILON_DECAY}")
    LOGGER.info(f"{colorstr('EPSILON_START:')} {EPSILON_START}    -    {colorstr('EPSILON_END:')} {EPSILON_END}")
    LOGGER.info(f"{colorstr('TARGET_UPDATE_FREQ:')} {TARGET_UPDATE_FREQ}")
    LOGGER.info(f"{colorstr('SAVE_PATH:')} {SAVE_PATH}")

    LOGGER.info(colorstr('black', 'bold', '%20s' + '%15s' * 4) % 
                        ('Training:', 'gpu_mem', 'AvgRew', 'AvgEpLen', 'Episodes'))
    
    greater_avg_rew = 0

    with tqdm(itertools.count(), total=100, 
            bar_format='{desc} {percentage:>7.0f}%|{bar:20}{r_bar}{bar:-10b}',
            unit='step') as pbar:
        for step in pbar:
            epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
            
            # rnd_sample = random.random()

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
                "Average Episode Length": len_mean,
                "Episodes": episode_count
            })

            if step % SAVE_INTERVAL == 0 and step != 0:
                LOGGER.info(f"\n{colorstr('black', 'bold', f'Saving model at {step}...')}")
                online_net.save(os.path.join(SAVE_PATH, f"{file_saveName}{step // SAVE_INTERVAL}0k.pack"))
                pbar.update(5)

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='base', type=str, help='select model: base, double, dueling')
    parser.add_argument('--env_id', default='ALE/Breakout-v5', type=str, help='enviroment id')
    parser.add_argument('--resume', default=False, type=bool, help="continue traning")
    parser.add_argument('--file_weight_path', type=str, help="pretrained weight path")
    parser.add_argument('--file_saveName', type=str, help="weight file name")
    parser.add_argument('--run_time', default=1, type=int, help="current run time")

    parser.add_argument('--wandb_project', default="Deep-Q-Learning-Network", type=str, help="wandb project name")
    parser.add_argument('--wandb_entity', default="devzxje", type=str, help="wandb username")
    parser.add_argument('--wandb_session', default="DQN-v1", type=str, help="wandb running title")
    parser.add_argument('--wandb_id', default="DQN-v1", type=str, help="wandb running id")
    parser.add_argument('--wandb_resume', default=None, help="wandb continue existed seasion_name executed")
    parser.add_argument('--wandb_relogin', default=False, type=bool, help="wandb force relogin")

    parser.add_argument('--device', default=device, help="select GPU or CPU for session")

    args = parser.parse_args()

    train(env_id=args.env_id, 
          resume= args.resume, 
          file_weight_path=args.file_weight_path, 
          file_saveName= args.file_saveName, 
          run_time=args.run_time,
          project=args.wandb_project,
          entity=args.wandb_entity,
          name=args.wandb_session,
          id_name=args.wandb_id,
          session_resume=args.wandb_resume,
          relogin=args.wandb_relogin,
          device=args.device)