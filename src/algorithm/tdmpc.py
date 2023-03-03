import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from torch.nn.functional import normalize

class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._action_encoder = h.mlp(cfg.action_dim, cfg.mlp_dim, cfg.latent_action_dim)
        #self._action_encoder = nn.Identity()
        self._action_decoder = h.mlp(cfg.latent_action_dim + cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._dynamics = h.mlp(cfg.latent_dim+cfg.latent_action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._inverse_dynamics = h.mlp(cfg.latent_dim*2, cfg.mlp_dim, cfg.latent_action_dim)
        self._reward = h.mlp(cfg.latent_dim+cfg.latent_action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.latent_action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, self._Q1, self._Q2]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def h(self, obs):
        """Encodes an observation into its latent representation (h)."""
        return self._encoder(obs)
    
    def f(self, act):
        """Encodes an observation into its latent representation (h)."""
        return self._action_encoder(act)

    def next(self, z, u):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, u], dim=-1)
        return self._dynamics(x), self._reward(x)

    # def reward(self, z, a):
    # 	x = torch.cat([z, a], dim=-1)
    # 	return self._org_reward(x)

    def prev_act(self, prev_z, z):
        """Predicts previous latent action"""
        x = torch.cat([prev_z, z], dim=-1)
        return self._inverse_dynamics(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, u):
        """Predict state-action value (Q)."""  
        x = torch.cat([z, u], dim=-1)
        return self._Q1(x), self._Q2(x)

    def decode_action(self, z, u):
        """Return Raw Action"""  
        x = torch.cat([z, u], dim=-1)
        return self._action_decoder(x)
        #return u

class TDMPC():
    """Implementation of TD-MPC learning + inference."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()
        self.pretrain_step = 0

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {'model': self.model.state_dict(),
                'model_target': self.model_target.state_dict()}

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d['model'])
        self.model_target.load_state_dict(d['model_target'])

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        G += discount * torch.min(*self.model.Q(z, self.model.pi(z, self.cfg.min_std)))
        return G

    @torch.no_grad()
    def plan(self, obs, eval_mode=False, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return self.model.decode_action(self.model.h(obs.squeeze(0)), torch.empty(self.cfg.latent_action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1))

        # Sample policy trajectories
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.cfg.latent_action_dim, device=self.device)
            z = self.model.h(obs).repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(z, self.cfg.min_std)
                z, _ = self.model.next(z, pi_actions[t])

        # Initialize state and parameters
        z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.cfg.latent_action_dim, device=self.device)
        std = 2*torch.ones(horizon, self.cfg.latent_action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, self.cfg.latent_action_dim, device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.latent_action_dim, device=std.device)
        return self.model.decode_action(self.model.h(obs.squeeze(0)), a)

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)

        # Loss is a weighted sum of Q-values
        pi_loss = 0
        for t,z in enumerate(zs):
            a = self.model.pi(z, self.cfg.min_std)
            Q = torch.min(*self.model.Q(z, a))
            pi_loss += -Q.mean() * (self.cfg.rho ** t)

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.pi_optim.step()
        self.model.track_q_grad(True)
        return pi_loss.item()

    @torch.no_grad()
    def _td_target(self, next_obs, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z = self.model.h(next_obs)
        td_target = reward + self.cfg.discount * \
            torch.min(*self.model_target.Q(next_z, self.model.pi(next_z, self.cfg.min_std)))
        return td_target

    def pretrain_encoder(self, replay_buffer, epochs=5, gradient_steps=20, writer=None):
        self.optim.zero_grad(set_to_none=True)
        self.model.train()
        for epoch in range(epochs):
            ### Train Action Encoder
            for m in [self.model._action_encoder, self.model._action_decoder]:
                h.set_requires_grad(m, True)
            h.set_requires_grad(self.model._encoder, False)
            for g in range(gradient_steps):
                batch = replay_buffer.sample_batch()
                reward_loss, consistency_loss, inv_consistency_loss, action_reconstruct_loss = self.train_latent_model(batch, writer=writer, step=self.pretrain_step)
                self.pretrain_step += 1
            
            ### Train State Encoder
            for m in [self.model._action_encoder, self.model._action_decoder]:
                h.set_requires_grad(m, False)
            h.set_requires_grad(self.model._encoder, True)
            for g in range(gradient_steps):
                batch = replay_buffer.sample_batch()
                reward_loss, consistency_loss, inv_consistency_loss, action_reconstruct_loss = self.train_latent_model(batch, writer=writer, step=self.pretrain_step)
                self.pretrain_step += 1
        
        ### Make sure encoder/decoders requires gradient
        for m in [self.model._action_encoder, self.model._action_decoder]:
                h.set_requires_grad(m, True)
        h.set_requires_grad(self.model._encoder, True)
        self.model.eval()
                

    def train_latent_model(self, batch, writer=None, step=None):
        
        self.optim.zero_grad(set_to_none=True)
        self.model.train()
        obs, action, reward, next_obs = batch
        u = self.model.f(action)
        curr_z, next_z =self.model.h(self.aug(obs)), self.model.h(self.aug(next_obs))
        z = self.model.h(obs)
        pred_next_z, reward_pred = self.model.next(curr_z, u)
        
        ### Forward Prediction
        reward_loss = torch.mean(h.mse(reward, reward_pred))
        consistency_loss = torch.mean(h.mse(pred_next_z, next_z))

        ### Inverse Prediction
        pred_u = self.model.prev_act(curr_z, next_z)
        inv_consistency_loss = torch.mean(h.mse(pred_u, u))

        ### Action Reconstruction Loss
        action_reconstruct = self.model.decode_action(z, u)     
        action_reconstruct_loss = torch.mean(h.mse(action, action_reconstruct))
        
        loss = reward_loss + consistency_loss + inv_consistency_loss + action_reconstruct_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()
        
        if writer is not None:
            writer.add_scalar('Pretrain/action_reconstruction_loss', 
                                action_reconstruct_loss, step)
            writer.add_scalar('Pretrain/inverse_consistency_loss', 
                                inv_consistency_loss, step)
            writer.add_scalar('Pretrain/consistency_loss', 
                                consistency_loss, step)
            writer.add_scalar('Pretrain/reward_loss', 
                                reward_loss, step)
        
        return reward_loss, consistency_loss, inv_consistency_loss, action_reconstruct_loss

    def update(self, replay_buffer, step):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        obs, obses, next_obses, action, reward, idxs, weights = replay_buffer.sample()
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z = self.model.h(self.aug(obs))
        zs = [z.detach()]

        inv_consistency_loss, act_reconstruct_loss, consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0, 0, 0
        for t in range(self.cfg.horizon):

            # Predictions
            u = self.model.f(action[t])
            Q1, Q2 = self.model.Q(z, u)
            z, reward_pred = self.model.next(z, u)
            #org_reward_pred = self.model.reward(z, action[t])
            with torch.no_grad():                
                next_obs = self.aug(next_obses[t])
                next_z = self.model_target.h(next_obs)
                td_target = self._td_target(next_obs, reward[t])
                z1, z2 = self.model_target.h(obses[t]), self.model_target.h(next_obses[t])
            u_predict = self.model.prev_act(z1, z2)
            zs.append(z.detach())
            action_reconstruct = self.model.decode_action(z, u)

            # Losses
            rho = (self.cfg.rho ** t)
            consistency_loss += rho * torch.mean(h.mse(z, next_z), dim=1, keepdim=True)
            inv_consistency_loss += rho * torch.mean(h.mse(u_predict, u), dim=1, keepdim=True)
            act_reconstruct_loss += rho * torch.mean(h.mse(action[t], action_reconstruct), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            value_loss += rho * (h.mse(Q1, td_target) + h.mse(Q2, td_target))
            priority_loss += rho * (h.l1(Q1, td_target) + h.l1(Q2, td_target))
        # Optimize model
        total_loss = self.cfg.act_reconstruct_coef * act_reconstruct_loss.clamp(max=1e4) + \
                    self.cfg.consistency_coef * consistency_loss.clamp(max=1e4) + \
                    self.cfg.inv_consistency_coef * inv_consistency_loss.clamp(max=1e4) + \
                    self.cfg.reward_coef * reward_loss.clamp(max=1e4) + \
                    self.cfg.value_coef * value_loss.clamp(max=1e4) \

        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1/self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False)
        self.optim.step()
        replay_buffer.update_priorities(idxs, priority_loss.clamp(max=1e4).detach())

        # Update policy + target network
        pi_loss = self.update_pi(zs)
        if step % self.cfg.update_freq == 0:
            h.ema(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {'consistency_loss': float(consistency_loss.mean().item()),
                'inverse_consistency_loss': float(inv_consistency_loss.mean().item()),
                'action_reconstruction_loss': float(act_reconstruct_loss.mean().item()),
                'reward_loss': float(reward_loss.mean().item()),
                'value_loss': float(value_loss.mean().item()),
                'pi_loss': pi_loss,
                'total_loss': float(total_loss.mean().item()),
                'weighted_loss': float(weighted_loss.mean().item()),
                'grad_norm': float(grad_norm),
                }