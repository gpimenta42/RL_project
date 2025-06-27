
# --------------------------------------------------------
# Adapted from:
# https://github.com/Curt-Park/rainbow-is-all-you-need
# --------------------------------------------------------

import sys

import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
import pandas as pd
# !curl -O https://raw.githubusercontent.com/curt-park/rainbow-is-all-you-need/master/segment_tree.py
# os.getcwd()
from utils.segment_tree import MinSegmentTree, SumSegmentTree



class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size



class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
    
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    
        
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
        )
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()

class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        learning_rate: float = 0.0001,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    
    def evaluate(self, num_episodes: int = 5, final_evaluate=False) -> float:
        """Evaluate current policy without exploration."""
        self.is_test = True
        
        if final_evaluate:
            self.dqn.load_state_dict(torch.load("best_model.pth", map_location=self.device))
            self.dqn.eval() 
        
        scores = []

        for _ in range(num_episodes):
            state, _ = self.env.reset(seed=None)
            score = 0
            done = False
            while not done:
                action = self.select_action(state)
                state, reward, done = self.step(action)
                score += reward
            scores.append(score)

        self.is_test = False
        return np.mean(scores), np.std(scores)

    
    def train(self, num_frames: int, plotting_interval: int = 10_000):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset()
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        best_score = -float('inf')
        
        eval_means = []
        eval_stds = []  
        eval_frames = []
        
        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state, _ = self.env.reset()
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                eval_score, std = self.evaluate(num_episodes=10)
                eval_means.append(eval_score)
                eval_stds.append(std)
                eval_frames.append(frame_idx)
                print(f"Eval score at frame {frame_idx}: {eval_score:.2f}")
                if eval_score > best_score:
                    best_score = eval_score
                    torch.save(self.dqn.state_dict(), "best_model.pth")
                    print("New best model saved.")
                self._plot(eval_frames, eval_means, eval_stds, losses)
        
        results_df = pd.DataFrame({
            'frame': eval_frames,
            'mean_return': eval_means,
            'std_return': eval_stds
        })
        results_df.to_csv("evaluation_results.csv", index=False)
        self.env.close()
                
    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True
        
        self.dqn.load_state_dict(torch.load("best_model.pth"))
        self.dqn.eval()

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
        
        state, _ = self.env.reset()
        done = False
        score = 0
        
        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        # reset
        self.env = naive_env


    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.floor().long() + 1

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u.clamp(max=self.atom_size - 1) + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        eval_frames: List[int],
        eval_means: List[float],
        eval_stds: List[float],
        losses: List[float],
    ):
        """Plot the training progresses."""
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title("Evaluation Return (Mean ± Std)")
        plt.plot(eval_frames, eval_means)
        plt.fill_between(
            eval_frames,
            np.array(eval_means) - np.array(eval_stds),
            np.array(eval_means) + np.array(eval_stds),
            alpha=0.3,
        )
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
        


import base64
import glob
import io
import os

from IPython.display import HTML, display


def ipython_show_video(path: str) -> None:
    """Show a video at `path` within IPython Notebook."""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()
    encoded = base64.b64encode(video)

    display(HTML(
        data="""
        <video width="320" height="240" alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
        </video>
        """.format(encoded.decode("ascii"))
    ))


def show_latest_video(video_folder: str) -> str:
    """Show the most recently recorded video from video folder."""
    list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    latest_file = max(list_of_files, key=os.path.getctime)
    ipython_show_video(latest_file)
    return latest_file


import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage

# Save path
def get_unique_log_dir(base_dir, env_name):
    i = 0
    while True:
        log_dir = os.path.join(base_dir, f"{env_name}_model_{i}")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            return log_dir
        i += 1
        
# Create the environment
def create_env(env_id, n_envs=1, wrapper_class=WarpFrame, n_stack=4, seed=None):
    env = make_vec_env(env_id, n_envs=n_envs, wrapper_class=wrapper_class, seed=seed)
    env = VecFrameStack(env, n_stack)
    env = VecTransposeImage(env)
    return env

# Create Callback

def create_eval_callback(env, best_model_save_path, log_path, eval_freq=25000, render=False, n_eval_episodes=20):
    return EvalCallback(
        eval_env=env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        render=render,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    
    

from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import os

def discretize_env(env, DISCRETE_BUCKETS):
    """
        This function discretizes the environment into bins sectioned across each part specified in env.observation_space

        env - the environment to bin
        num_bins - the number of bins per each part of the state (i.e. x_pos)
    """
    obs_low = env.observation_space.low
    obs_high = env.observation_space.high
    
    bins = {}
    names = [
        "x_pos", "y_pos", "x_vel", "y_vel",
        "angle", "ang_vel", "left_leg", "right_leg"
    ]
    
    for i, name in enumerate(names):
        if name in ["left_leg", "right_leg"]:
            # Binary: no need for bins
            bins[name] = [0, 1]
        else:
            bins[name] = np.linspace(obs_low[i], obs_high[i], DISCRETE_BUCKETS + 1)
    
    return bins


def discretize(obs, bins):
    """
        This function returns each part of the state for the current observation

        obs - the observation returned by the environment
        bins - the discretized state
    """

    x, y, x_dot, y_dot, angle, ang_vel, left, right = obs

    state = (
        np.digitize(x, bins["x_pos"]) - 1,
        np.digitize(y, bins["y_pos"]) - 1,
        np.digitize(x_dot, bins["x_vel"]) - 1,
        np.digitize(y_dot, bins["y_vel"]) - 1,
        np.digitize(angle, bins["angle"]) - 1,
        np.digitize(ang_vel, bins["ang_vel"]) - 1,
        int(left),
        int(right)
    )
    return state


def epsilon_greedy_policy(env, state, Q_dict, epsilon=.5):
    
    #sample a random value from the uniform distribution, if the sampled value is less than
    #epsilon then we select a random action else we select the best action which has maximum Q
    #value as shown below
    
    if np.random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q_dict[(state,x)])
    
def Q_table_epsilon_greedy_policy(env, state, actio_state_table, epsilon=.5):
    
    #sample a random value from the uniform distribution, if the sampled value is less than
    #epsilon then we select a random action else we select the best action which has maximum Q
    #value as shown below
    
    if np.random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(actio_state_table[state])

def generate_episode(env, epsilon_greedy_policy, Q_dict, num_timestep=200, epsilon=.5, render=False):
    
    #let's define a list called episode for storing the episode
    episode = []
    
    #initialize the state by resetting the environment
    state = env.reset()[0]
    
    #then for each time step
    for i in range(num_timestep):
        
        #select the action according to the given policy
        action = epsilon_greedy_policy(env, state, Q_dict, epsilon=epsilon)
        
        #perform the action and store the next state information
        next_state, reward, done, truncated, info = env.step(action)

        #store the state, action, reward into our episode list
        episode.append((state, action, reward))
        
        
        #If the next state is a final state then break the loop else update the next state to the current state
        if done:
            break
            
        state = next_state

        if render:
            time.sleep(.5)
            print("STEP:", i, "in state:", state, "with action", action, "with reward", reward)
    return episode

def default_q_values():
    return np.random.uniform(-1, 0, 4)

def SARSA_control(env, num_iteractions=100, alpha=0.85, gamma=0.9, epsilon=1.0, decay=1/100, bins_per_feature=10,
                  eval_step_interval=1_000, n_eval_episodes=10):

    Q_table = defaultdict(default_q_values)
    min_epsilon = 0.01
    return_over_time = []

    eval_means = []
    eval_stds = []
    eval_steps = []

    bins = discretize_env(env, bins_per_feature)

    total_steps = 0
    next_eval_step = eval_step_interval

    for i in tqdm(range(num_iteractions), desc="SARSA Training"):
        state = env.reset()[0]
        state_discrete = discretize(state, bins)
        action = Q_table_epsilon_greedy_policy(env, state_discrete, Q_table, epsilon)

        num_timesteps = 1_000

        Return = 0
        # for t in range(400):
        for t in range(num_timesteps):
            next_state, r, done, truncated, _ = env.step(action)
            next_state_discrete = discretize(next_state, bins)
            next_action = Q_table_epsilon_greedy_policy(env, next_state_discrete, Q_table, epsilon)

            td_target = r + gamma * Q_table[next_state_discrete][next_action]
            td_error = td_target - Q_table[state_discrete][action]
            Q_table[state_discrete][action] += alpha * td_error

            state_discrete = next_state_discrete
            action = next_action

            Return += r
            total_steps += 1
            epsilon = max(min_epsilon, epsilon - decay)

            # --- Evaluate every `eval_step_interval` steps ---
            if total_steps >= next_eval_step:
                eval_returns = []
                for _ in range(n_eval_episodes):
                    eval_state = env.reset()[0]
                    eval_state_discrete = discretize(eval_state, bins)
                    eval_return = 0

                    for _ in range(400):
                        eval_action = np.argmax(Q_table[eval_state_discrete])  # greedy
                        next_eval_state, r, done, truncated, _ = env.step(eval_action)
                        eval_state_discrete = discretize(next_eval_state, bins)
                        eval_return += r
                        if done:
                            break

                    eval_returns.append(eval_return)

                eval_means.append(np.mean(eval_returns))
                eval_stds.append(np.std(eval_returns))
                eval_steps.append(total_steps)
                next_eval_step += eval_step_interval

            if done:
                break

        return_over_time.append([i, Return])

    # --- Plotting (Mean ± Std Dev) ---
    plt.figure(figsize=(10, 6))
    plt.plot(eval_steps, eval_means, label="Mean Evaluation Return")
    plt.fill_between(
        eval_steps,
        np.array(eval_means) - np.array(eval_stds),
        np.array(eval_means) + np.array(eval_stds),
        alpha=0.3,
        label="±1 Std Dev"
    )
    plt.xlabel("Steps")
    plt.ylabel("Evaluation Return")
    plt.title("SARSA Evaluation: Mean ± Std Dev (per 10k steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("SARSA_reward_distribution.png")
    plt.close()
    # plt.show()

    df_return_over_time = pd.DataFrame(return_over_time, columns=["Episode", "Return"])
    return Q_table, df_return_over_time

# def SARSA_control(env, num_iteractions=100, alpha=0.85, gamma=0.9, epsilon=1.0, decay=1/100, bins_per_feature=10):

    # Q_table = defaultdict(lambda: np.random.uniform(-1, 0, env.action_space.n))
    Q_table = defaultdict(default_q_values)
    min_epsilon = 0.01
    return_over_time = []

    bins = discretize_env(env, bins_per_feature)
    
    for i in tqdm(range(num_iteractions), desc="SARSA Training"):
        state = env.reset()[0]
        state_discrete = discretize(state, bins)
        action = Q_table_epsilon_greedy_policy(env, state_discrete, Q_table, epsilon)

        Return = 0
        for t in range(400):
            next_state, r, done, truncated, info = env.step(action)
            next_state_discrete = discretize(next_state, bins)
            next_action = Q_table_epsilon_greedy_policy(env, next_state_discrete, Q_table, epsilon)

            # SARSA update rule
            td_target = r + gamma * Q_table[next_state_discrete][next_action]
            td_error = td_target - Q_table[state_discrete][action]
            Q_table[state_discrete][action] += alpha * td_error

            state_discrete = next_state_discrete
            action = next_action
            Return += r
            epsilon = max(min_epsilon, epsilon - decay)

            if done:
                break

        return_over_time.append([i, Return])
    
    df_return_over_time = pd.DataFrame(return_over_time, columns=["Episode", "Return"])
    return Q_table, df_return_over_time

def Q_learning(env, num_iteractions=300, alpha=.85, gamma=.9, epsilon=1.0, decay=1/100, bins_per_feature=10,
               eval_interval=1000, n_eval_episodes=10):

    Q_table = defaultdict(default_q_values)
    min_epsilon = 0.01
    return_over_time = []

    eval_means = []
    eval_stds = []
    eval_episodes = []

    bins = discretize_env(env, bins_per_feature)

    for i in tqdm(range(num_iteractions)):
        state = env.reset()[0]
        state_discrete = discretize(state, bins)

        num_timesteps = 1_000
        Return = 0

        for t in range(num_timesteps):
            action = Q_table_epsilon_greedy_policy(env, state_discrete, Q_table, epsilon=epsilon)
            next_state, r, done, truncated, _ = env.step(action)
            next_state_discrete = discretize(next_state, bins)

            best_next_action = np.argmax(Q_table[next_state_discrete])
            td_target = r + gamma * Q_table[next_state_discrete][best_next_action]
            td_error = td_target - Q_table[state_discrete][action]
            Q_table[state_discrete][action] += alpha * td_error

            state_discrete = next_state_discrete
            epsilon = max(min_epsilon, epsilon - decay)

            Return += r
            if done:
                break

        return_over_time.append([i, Return])

        # --- Evaluation every eval_interval episodes ---
        if (i + 1) % eval_interval == 0:
            eval_returns = []
            for _ in range(n_eval_episodes):
                state = env.reset()[0]
                state_discrete = discretize(state, bins)
                eval_return = 0

                for _ in range(num_timesteps):
                    action = np.argmax(Q_table[state_discrete])  # greedy action
                    next_state, r, done, truncated, _ = env.step(action)
                    state_discrete = discretize(next_state, bins)
                    eval_return += r
                    if done:
                        break

                eval_returns.append(eval_return)

            eval_means.append(np.mean(eval_returns))
            eval_stds.append(np.std(eval_returns))
            eval_episodes.append(i + 1)

    # --- Plot Mean ± Std ---
    plt.figure(figsize=(10, 6))
    plt.plot(eval_episodes, eval_means, label="Mean Evaluation Return")
    plt.fill_between(
        eval_episodes,
        np.array(eval_means) - np.array(eval_stds),
        np.array(eval_means) + np.array(eval_stds),
        alpha=0.3,
        label="±1 Std Dev"
    )
    plt.xlabel("Episode")
    plt.ylabel("Evaluation Return")
    plt.title("Q-Learning Evaluation: Mean ± Std Dev")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Q-Learning_reward_distribution.png")
    plt.close()
    # plt.show()

    df_return_over_time = pd.DataFrame(return_over_time, columns=['Episode', 'Return'])

    return Q_table, df_return_over_time



# def Q_learning(env, num_iteractions=100, alpha=.85, gamma=.9, epsilon=1.0, decay=1/100, bins_per_feature=10):

    #Initiallize Q_table as numpy array
    #Q_table = np.zeros((env.observation_space.n,env.action_space.n))

    #Initialize with random values
    Q_table = defaultdict(default_q_values)

    #Initialize Epsilon Decay
    min_epsilon = 0.01
    return_over_time = []
    bins = discretize_env(env, bins_per_feature)
    
    #for each episode
    for i in tqdm(range(num_iteractions)):
        
        #initialize the state by resetting the environment
        state = env.reset()[0]
        state_discrete = discretize(state, bins)    
        
        num_timesteps = 300
        
        Return = 0
        
        #for each step in the episode:
        for t in range(num_timesteps):
            
            #select the action using the epsilon-greedy policy
            action = Q_table_epsilon_greedy_policy(env, state_discrete, Q_table, epsilon=epsilon)

            #perform the selected action and store the next state information: 
            next_state, r, done, truncated, _ = env.step(action)

            # state_discrete = discretize(state, bins)
            next_state_discrete = discretize(next_state, bins)

            #compute the Q value of the state-action pair
            best_next_action = np.argmax(Q_table[next_state_discrete])
            td_target = r + gamma * Q_table[next_state_discrete][best_next_action]
            td_error = td_target - Q_table[state_discrete][action]
            Q_table[state_discrete][action] += alpha * td_error

            #update next state to current state
            state_discrete = next_state_discrete
            epsilon = max(min_epsilon, epsilon - decay)
            
            Return += r
            
            #if the current state is the terminal state then break:
            if done:
                break

        return_over_time.append([i, Return])
    
    df_return_over_time = pd.DataFrame(return_over_time, columns=['Episode', 'Return'])
    
    return Q_table, df_return_over_time

def visualize_results(df_return_over_time, Q_table, bins_per_feature=10, output_dir="results", algorithm="Q-Learning"):
    os.makedirs(output_dir, exist_ok=True)

    # Plot raw episode returns
    plt.figure(figsize=(10, 5))
    plt.plot(df_return_over_time["Episode"], df_return_over_time["Return"], alpha=0.5, label="Episode Return")
    plt.title(algorithm + " Raw Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "raw_returns"+algorithm+".png"))
    plt.close()

    # Plot smoothed returns
    plt.figure(figsize=(10, 5))
    df_return_over_time["Smoothed"] = df_return_over_time["Return"].rolling(window=100).mean()
    plt.plot(df_return_over_time["Episode"], df_return_over_time["Smoothed"], color="green", label="Smoothed (100 ep)")
    plt.title(algorithm + " Smoothed Return per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "smoothed_returns"+algorithm+".png"))
    plt.close()

    # Auto-detect Q-table format and flatten if needed
    flat_Q = {}
    try:
        # Detect if already in flat form
        any_key = next(iter(Q_table))
        if isinstance(any_key, tuple) and isinstance(any_key[0], tuple):
            # Format: Q_table[(state_tuple, action)] = value
            flat_Q = Q_table
        else:
            # Assume nested form: Q_table[state_tuple][action] = value
            for state, action_vals in Q_table.items():
                for action, value in enumerate(action_vals):
                    flat_Q[(state, action)] = value
    except Exception as e:
        print("Error flattening "+algorithm+" Q_table:", e)
        return

    # Detect number of actions from Q_table
    try:
        num_actions = max(a for (_, a) in flat_Q.keys()) + 1
    except:
        num_actions = 4  # fallback

    # Create heatmap of max Q-values
    try:
        q_vals = np.full((bins_per_feature, bins_per_feature), -np.inf)
        for (state, action), value in flat_Q.items():
            if isinstance(state, tuple) and len(state) >= 2:
                x, y = state[0], state[1]
                if 0 <= x < bins_per_feature and 0 <= y < bins_per_feature:
                    q_vals[x, y] = max(q_vals[x, y], value)

        plt.figure(figsize=(6, 5))
        sns.heatmap(q_vals, cmap="viridis", annot=False)
        plt.title(algorithm + " Max Q-values (first 2 state dims)")
        plt.xlabel("State Dim 1")
        plt.ylabel("State Dim 0")
        plt.savefig(os.path.join(output_dir, "q_value_heatmap"+algorithm+".png"))
        plt.close()
    except Exception as e:
        print("Skipped "+algorithm+" heatmap:", e)

    # Create policy map (best action per state)
    try:
        policy_map = np.full((bins_per_feature, bins_per_feature), -1)
        for (state, _), _ in flat_Q.items():
            if isinstance(state, tuple) and len(state) >= 2:
                x, y = state[0], state[1]
                if 0 <= x < bins_per_feature and 0 <= y < bins_per_feature:
                    actions = [flat_Q.get((state, a), -np.inf) for a in range(num_actions)]
                    best_action = int(np.argmax(actions))
                    policy_map[x, y] = best_action

        plt.figure(figsize=(6, 5))
        plt.imshow(policy_map, cmap='tab10', interpolation='nearest')
        plt.colorbar(label=algorithm + " Best Action")
        plt.title(algorithm + " Policy Map (Best Action by State)")
        plt.xlabel("State Dim 1")
        plt.ylabel("State Dim 0")
        plt.savefig(os.path.join(output_dir, "policy_map.png"))
        plt.close()
    except Exception as e:
        print("Skipped "+algorithm+" policy map:", e)




import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        best_model_save_path=None,
        log_path=None,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render = render
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path

        self.eval_means = []
        self.eval_stds = []
        self.timesteps = []
        self.best_mean_reward = -float("inf")

        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            self.log_file = open(os.path.join(self.log_path, "eval_log.csv"), "w")
            self.log_file.write("timesteps,mean_reward,std_reward\n")

        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True
            )
            mean = np.mean(rewards)
            std = np.std(rewards)

            self.eval_means.append(mean)
            self.eval_stds.append(std)
            self.timesteps.append(self.num_timesteps)

            # Save best model
            if mean > self.best_mean_reward:
                self.best_mean_reward = mean
                if self.best_model_save_path:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))

            # Log to CSV
            if self.log_path:
                self.log_file.write(f"{self.num_timesteps},{mean},{std}\n")
                self.log_file.flush()

            if self.verbose > 0:
                print(f"Eval @ {self.num_timesteps}: {mean:.2f} ± {std:.2f}")

        return True

    def _on_training_end(self) -> None:
        if self.log_path:
            self.log_file.close()