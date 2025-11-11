"""
Evaluation script that captures attention weights for visualization.
"""
from datetime import datetime
from glob import glob
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import torch.nn.functional as F
import os.path as path
import numpy as np
from einops import pack, rearrange

from env import SAMPLE_ENVIRONMENT, make_env, map_dark_states
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class AttentionCapture:
    """Helper class to capture attention weights during forward pass."""
    
    def __init__(self):
        self.attention_weights = []
        self.hooks = []
    
    def register_hooks(self, model):
        """Register forward hooks on transformer encoder layers."""
        for layer_idx, layer in enumerate(model.transformer_encoder.layers):
            hook = self._make_hook(layer_idx)
            handle = layer.self_attn.register_forward_hook(hook)
            self.hooks.append(handle)
    
    def _make_hook(self, layer_idx):
        """Create a hook that captures attention weights."""
        def hook(module, input, output):
            # During forward, we need to manually compute attention to get weights
            # The MultiheadAttention module doesn't return weights by default
            # We'll need to call it with need_weights=True
            pass
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights = []


def compute_attention_weights(model, x, seq_len):
    """
    Manually compute attention weights by accessing the transformer internals.
    
    Args:
        model: The AD model
        x: Input embeddings (batch, seq, emb)
        seq_len: Sequence length
    
    Returns:
        List of attention weights for each layer
    """
    attention_weights_all_layers = []
    
    # Apply positional embedding
    x = x + model.pos_embedding[:, :seq_len, :]
    
    # Pass through each transformer encoder layer and capture attention
    for layer in model.transformer_encoder.layers:
        # Get attention weights from self-attention
        # We need to call the attention module directly with need_weights=True
        attn_output, attn_weights = layer.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False  # Keep per-head attention
        )
        
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)
        attention_weights_all_layers.append(attn_weights.detach().cpu().numpy())
        
        # Continue with the layer's forward pass
        x = layer.norm1(x + layer.dropout1(attn_output))
        x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
        x = layer.norm2(x + layer.dropout2(x2))
    
    return attention_weights_all_layers


def evaluate_with_attention_capture(model, vec_env, eval_timesteps, config):
    """
    Evaluate the model while capturing attention weights.
    
    Args:
        model: The AD model
        vec_env: Vectorized environment
        eval_timesteps: Number of timesteps to evaluate
        config: Configuration dictionary
    
    Returns:
        Dictionary with rewards and attention history
    """
    outputs = {}
    outputs['reward_episode'] = []
    outputs['attention_history'] = []  # Store attention patterns over time
    
    reward_episode = np.zeros(vec_env.num_envs)
    
    query_states = vec_env.reset()
    query_states = torch.tensor(query_states, device=device, requires_grad=False, dtype=torch.long)
    query_states = rearrange(query_states, 'e d -> e 1 d')
    query_states_embed = model.embed_query_state(map_dark_states(query_states, model.grid_size))
    transformer_input = query_states_embed
    
    for step in range(eval_timesteps):
        query_states_prev = query_states.clone().detach().to(torch.float)
        
        # Compute transformer output and capture attention
        seq_len = transformer_input.size(1)
        attention_weights = compute_attention_weights(model, transformer_input, seq_len)
        
        # Store attention for this timestep
        outputs['attention_history'].append(attention_weights)
        
        # Get action predictions (recompute the final forward pass)
        output = model.transformer(transformer_input,
                                   max_seq_length=model.max_seq_length,
                                   dtype='fp32')
        logits = model.pred_action(output[:, -1])
        
        # Sample action
        log_probs = F.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs.exp(), num_samples=1)
        actions = rearrange(actions, 'e 1 -> e')
        
        # Step environment
        query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
        
        actions = rearrange(actions, 'e -> e 1 1')
        actions = F.one_hot(actions, num_classes=config['num_actions'])
        
        reward_episode += rewards
        rewards = torch.tensor(rewards, device=device, requires_grad=False, dtype=torch.float)
        rewards = rearrange(rewards, 'e -> e 1 1')
        
        query_states = torch.tensor(query_states, device=device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        
        if dones[0]:
            outputs['reward_episode'].append(reward_episode)
            reward_episode = np.zeros(vec_env.num_envs)
            
            states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]),
                                       device=device, dtype=torch.float)
            states_next = rearrange(states_next, 'e d -> e 1 d')
        else:
            states_next = query_states.clone().detach().to(torch.float)
        
        query_states_embed = model.embed_query_state(map_dark_states(query_states, model.grid_size))
        
        # Build next transformer input
        context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
        context_embed = model.embed_context(context)
        
        if transformer_input.size(1) > 1:
            context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
            context_embed = context_embed[:, -(model.n_transit-1):]
        
        transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')
        
        # Print progress
        if (step + 1) % 50 == 0:
            print(f"Step {step + 1}/{eval_timesteps}, Seq len: {transformer_input.size(1)}")
    
    outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
    
    return outputs


if __name__ == '__main__':
    ckpt_dir = './runs/AD-darkroom-seed0'
    output_dir = './figs/attention_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print("Evaluation goals: ", test_env_args)

    if env_name == 'darkroom':
        envs = SubprocVecEnv([make_env(config, goal=test_env_args[7])])
    else:
        raise NotImplementedError(f'Environment not supported')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting attention-aware evaluation at {start_time}')

    # Evaluate with attention capture
    # Use a reasonable number of episodes for analysis (not just timesteps)
    # Typical episode length is config['horizon'], so run multiple episodes
    eval_episodes = 500  
    eval_timesteps = config['horizon'] * eval_episodes
    
    print(f"Running evaluation for {eval_episodes} episodes (~{eval_timesteps} steps)")
    print(f"Episode horizon: {config['horizon']} steps\n")
    
    with torch.no_grad():
        results = evaluate_with_attention_capture(
            model, envs, 
            eval_timesteps=eval_timesteps,
            config=config
        )
    
    end_time = datetime.now()
    print()
    print(f'Ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    envs.close()

    # Save results
    # Note: attention_history contains variable-length sequences, so we save it separately
    result_path = path.join(output_dir, 'attention_data.npz')
    
    # Save rewards as regular array
    np.savez(result_path, rewards=results['reward_episode'])
    
    # Save attention history as a pickle file (handles variable-length sequences)
    attention_path = path.join(output_dir, 'attention_history.pkl')
    import pickle
    with open(attention_path, 'wb') as f:
        pickle.dump(results['attention_history'], f)
    
    print(f"\nSaved attention data to {result_path}")
    print(f"Saved attention history to {attention_path}")
    
    print("\nRewards:")
    print(f"  Mean reward: {results['reward_episode'].mean():.2f}")
    print(f"  Std deviation: {results['reward_episode'].std():.2f}")
    
    print(f"\nCaptured attention for {len(results['attention_history'])} timesteps")
    print(f"  Number of layers: {len(results['attention_history'][0])}")
    if len(results['attention_history'][0]) > 0:
        print(f"  Number of heads: {results['attention_history'][0][0].shape[1]}")
        print(f"  Initial sequence length: {results['attention_history'][0][0].shape[2]}")
        print(f"  Final sequence length: {results['attention_history'][-1][0].shape[2]}")
    
    print("\nNext steps:")
    print("  Run: python analyze_attention.py")
    print("  This will generate visualizations for compression strategy.")
