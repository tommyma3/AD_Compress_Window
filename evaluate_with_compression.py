"""
Evaluate Algorithm Distillation model with compression enabled.

This script evaluates the model with uniform compression triggered when
max sequence length is reached during evaluation.
"""

import os
import numpy as np
import torch
from glob import glob

from model import MODEL
from utils import get_config
from env import SAMPLE_ENVIRONMENT, make_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import multiprocessing


def evaluate_with_compression(config, enable_compression=True, compression_ratio=0.5):
    """
    Evaluate model with optional compression.
    
    Args:
        config: Configuration dictionary
        enable_compression: Whether to enable compression
        compression_ratio: Compression ratio (0.0 to 1.0)
    """
    # Find latest checkpoint
    log_dir = f"./runs/AD-darkroom-seed0"
    ckpt_paths = sorted(glob(os.path.join(log_dir, 'ckpt-*.pt')))
    
    if not ckpt_paths:
        print(f"Error: No checkpoint found in {log_dir}")
        return None
    
    ckpt_path = ckpt_paths[-1]
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    config['enable_compression'] = enable_compression
    config['compression_ratio'] = compression_ratio
    
    print(f"Using device: {device}")
    print(f"Compression enabled: {enable_compression}")
    if enable_compression:
        print(f"Compression ratio: {compression_ratio}")
    
    model = MODEL[config['model']](config)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    # Create evaluation environments
    env_name = config['env']
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    eval_env_args = test_env_args  # Use all test environments
    
    print(f"\nCreating {len(eval_env_args)} evaluation environments...")
    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in eval_env_args])
    else:
        raise NotImplementedError('Environment not supported')
    
    # Evaluate
    eval_timesteps = config['horizon'] * 500
    print(f"\nEvaluating for {eval_timesteps} timesteps...")
    print("="*60)
    
    with torch.no_grad():
        outputs = model.evaluate_in_context(envs, eval_timesteps)
    
    # Process results
    reward_episodes = outputs['reward_episode']
    mean_rewards = reward_episodes.mean(axis=1)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of environments: {len(eval_env_args)}")
    print(f"Evaluation timesteps: {eval_timesteps}")
    print(f"\nMean reward per environment:")
    for i, (goal, reward) in enumerate(zip(eval_env_args, mean_rewards)):
        print(f"  Env {i} (goal {goal}): {reward:.3f}")
    print(f"\nOverall mean reward: {mean_rewards.mean():.3f}")
    print(f"Std deviation: {mean_rewards.std():.3f}")
    print(f"Min: {mean_rewards.min():.3f}, Max: {mean_rewards.max():.3f}")
    
    envs.close()
    
    return {
        'mean_reward': mean_rewards.mean(),
        'std_reward': mean_rewards.std(),
        'per_env_rewards': mean_rewards
    }


def compare_compression_modes(config):
    """Compare performance with and without compression."""
    print("\n" + "="*70)
    print("COMPARING COMPRESSION VS NO COMPRESSION")
    print("="*70)
    
    # Evaluate without compression
    print("\n" + "="*70)
    print("MODE 1: NO COMPRESSION (Baseline)")
    print("="*70)
    results_no_comp = evaluate_with_compression(config, enable_compression=False)
    
    # Evaluate with compression
    print("\n" + "="*70)
    print("MODE 2: WITH COMPRESSION (Uniform Sampling, 50% ratio)")
    print("="*70)
    results_with_comp = evaluate_with_compression(config, enable_compression=True, 
                                                   compression_ratio=0.5)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"No Compression:   Mean Reward = {results_no_comp['mean_reward']:.3f} ± {results_no_comp['std_reward']:.3f}")
    print(f"With Compression: Mean Reward = {results_with_comp['mean_reward']:.3f} ± {results_with_comp['std_reward']:.3f}")
    
    diff = results_with_comp['mean_reward'] - results_no_comp['mean_reward']
    pct_change = (diff / results_no_comp['mean_reward']) * 100
    
    print(f"\nDifference: {diff:+.3f} ({pct_change:+.1f}%)")
    
    if abs(pct_change) < 5:
        print("✓ Compression has minimal impact on performance!")
    elif pct_change < -5:
        print("⚠ Compression degrades performance. Consider:")
        print("  - Using different compression ratio")
        print("  - Using attention-based compression")
        print("  - Keeping more recent history")
    else:
        print("✓ Compression improves performance (possibly due to regularization)")
    
    return results_no_comp, results_with_comp


def test_different_ratios(config):
    """Test different compression ratios."""
    print("\n" + "="*70)
    print("TESTING DIFFERENT COMPRESSION RATIOS")
    print("="*70)
    
    ratios = [0.3, 0.5, 0.7]
    results = {}
    
    for ratio in ratios:
        print(f"\n" + "="*70)
        print(f"COMPRESSION RATIO: {ratio:.1%}")
        print("="*70)
        results[ratio] = evaluate_with_compression(config, enable_compression=True, 
                                                   compression_ratio=ratio)
    
    # Summary
    print("\n" + "="*70)
    print("COMPRESSION RATIO COMPARISON")
    print("="*70)
    print(f"{'Ratio':<10} {'Mean Reward':<15} {'Std Dev':<10}")
    print("-"*70)
    for ratio, result in results.items():
        print(f"{ratio:.1%}       {result['mean_reward']:.3f}          {result['std_reward']:.3f}")
    
    # Find best ratio
    best_ratio = max(results.keys(), key=lambda r: results[r]['mean_reward'])
    print(f"\n✓ Best compression ratio: {best_ratio:.1%} "
          f"(Mean Reward: {results[best_ratio]['mean_reward']:.3f})")
    
    return results


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Load configuration
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_dr.yaml'))
    
    print("="*70)
    print("EVALUATION WITH COMPRESSION")
    print("="*70)
    print(f"Model: {config['model']}")
    print(f"Environment: {config['env']}")
    print(f"Max sequence length (n_transit): {config['n_transit']}")
    
    # Choose evaluation mode
    mode = "compare"  # Options: "single", "compare", "ratios"
    
    if mode == "single":
        # Single evaluation with compression
        print("\n→ Mode: Single evaluation with compression")
        evaluate_with_compression(config, enable_compression=True, compression_ratio=0.5)
    
    elif mode == "compare":
        # Compare with and without compression
        print("\n→ Mode: Compare compression vs no compression")
        compare_compression_modes(config)
    
    elif mode == "ratios":
        # Test different compression ratios
        print("\n→ Mode: Test different compression ratios")
        test_different_ratios(config)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
