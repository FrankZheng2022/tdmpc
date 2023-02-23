import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer, FullReplayBuffer
import logger
from logger import make_dir
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
	"""Evaluate a trained agent and optionally save a video."""
	episode_rewards = []
	for i in range(num_episodes):
		obs, done, ep_reward, t = env.reset(), False, 0, 0
		if video: video.init(env, enabled=(i==0))
		while not done:
			action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
			obs, reward, done, _ = env.step(action.cpu().numpy())
			ep_reward += reward
			if video: video.record(env)
			t += 1
		episode_rewards.append(ep_reward)
		if video: video.save(env_step)
	return np.nanmean(episode_rewards)


def train(cfg):
	"""Training script for TD-MPC. Requires a CUDA-enabled device."""
	assert torch.cuda.is_available()
	set_seed(cfg.seed)
	if cfg.amlt == 1:
		log_dir    = os.path.join(os.environ['AMLT_OUTPUT_DIR'], cfg.exp_name)
		buffer_dir = os.path.join(os.environ['AMLT_OUTPUT_DIR'], cfg.exp_name, 'buffer')
		make_dir(buffer_dir)
	else:
		log_dir   =  Path().cwd() / __LOGS__ / cfg.exp_name  
		buffer_dir = Path().cwd() / __LOGS__ / cfg.exp_name / 'buffer'
		make_dir(buffer_dir)
	env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
	#full_buffer = FullReplayBuffer(max(cfg.max_buffer_size, cfg.train_steps), cfg.obs_shape, cfg.action_shape)
    
	# Run training
	L = logger.Logger(log_dir, cfg)
	episode_idx, start_time, eval_rewards = 0, time.time(), [-np.inf]
	for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):

		# Collect trajectory
		obs = env.reset()
		episode = Episode(cfg, obs)
		while not episode.done:
			action = agent.plan(obs, step=step, t0=episode.first)
			next_obs, reward, done, _ = env.step(action.cpu().numpy())
			pixel_obs = np.moveaxis(env.get_pixel(), -1, 0)
			#full_buffer.add(obs, action.cpu().numpy(), reward, next_obs, done, pixel_obs)
			obs = next_obs
			episode += (obs, action, reward, done)
		assert len(episode) == cfg.episode_length
		buffer += episode

		# Update model
		train_metrics = {}
		if step >= cfg.seed_steps:
			num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
			for i in range(num_updates):
				train_metrics.update(agent.update(buffer, step+i))

		# Log training episode
		episode_idx += 1
		env_step = int(step*cfg.action_repeat)
		common_metrics = {
			'episode': episode_idx,
			'step': step,
			'env_step': env_step,
			'total_time': time.time() - start_time,
			'episode_reward': episode.cumulative_reward}
		train_metrics.update(common_metrics)
		L.log(train_metrics, category='train')

		# Evaluate agent periodically
		if env_step % cfg.eval_freq == 0:
			eval_reward = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
			common_metrics['episode_reward'] = eval_reward
			L.log(common_metrics, category='eval')
			if eval_reward > np.max(eval_rewards):
				L.save(agent)
			eval_rewards.append(eval_reward) 
		#if full_buffer.idx%10000 == 0 and full_buffer.idx > 0:
			#full_buffer.save(buffer_dir)       

	L.finish(agent)
	print('Training completed successfully')


if __name__ == '__main__':
	train(parse_cfg(Path().cwd() / __CONFIG__))
