import os
import torch
import torch.nn as nn
import numpy as np
import random

from pytorch_wrappers import PytorchLazyFrames
from configs import GAMMA
import msgpack

def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten()
    )
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(
        cnn,
        nn.Linear(n_flatten, final_layer),
        nn.ReLU()
    )

    return out


class Network(nn.Module):
    def __init__(self, env, device):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device

        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(
            conv_net,
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs, epsilon):
        obses_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)
        return actions

    def criterion(self, transitions, target_net):
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions], dtype=np.float32)
        rews = np.asarray([t[2] for t in transitions], dtype=np.float32)
        dones = np.asarray([t[3] for t in transitions], dtype=np.float32)
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses, type=np.float32)
            new_obses = np.asarray(new_obses, type=np.float32)

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        # targets = r + gamma * target q vals * (1 - dones)
        target_q_values = target_net(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)
        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)
        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_file_path):
        if not os.path.isfile(load_file_path):
            raise FileNotFoundError(load_file_path)
            
        with open(load_file_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}

        self.load_state_dict(params)