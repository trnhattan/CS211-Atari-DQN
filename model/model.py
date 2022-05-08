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
    def __init__(self, env, device, model=None):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device
        self.model = model

        conv_net = nature_cnn(env.observation_space)

        if model == 'base' or model == 'double':
            self.net = nn.Sequential(
                conv_net,
                nn.Linear(512, self.num_actions)
            )

        elif model == 'dueling':
            pass

        else:
            raise NotImplementedError(f"{self.model} not found")

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
        with torch.no_grad():
            if self.model == 'double':
                targets_online_q_values = self(new_obses_t)
                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                targets_target_q_values = target_net(new_obses_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

                targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values

            elif self.model == 'dueling':
                targets_online_q_values = self(new_obses_t)
                targets_target_q_values = target_net(new_obses_t)

                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

                targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values

            elif self.model == 'base':
                target_q_values = target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

            else:
                raise NotImplementedError(f"{self.model} not found")

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

class DuelingDQN(Network):
    def __init__(self, env, device, model='dueling'):
        super().__init__(env, device, model)
        n_input_channels = env.observation_space.shape[0]
        depths = (32, 64, 64)
        final_layer = 512

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(env.observation_space.sample()[None]).float()).shape[1]

        self.fc_V = nn.Sequential(
            nn.Linear(n_flatten, final_layer),
            nn.ReLU()
        )
        
        self.fc_A = nn.Sequential(
            nn.Linear(n_flatten, final_layer),
            nn.ReLU()
        )

        self.V = nn.Linear(final_layer, 1)
        self.A = nn.Linear(final_layer, self.num_actions)

    def forward(self, x):
        cnn = self.cnn(x)

        V_out = self.fc_V(cnn)
        A_out = self.fc_A(cnn)

        V = self.V(V_out)
        A = self.A(A_out)

        Q = V + A - torch.mean(A, dim=1, keepdim=True)

        return Q