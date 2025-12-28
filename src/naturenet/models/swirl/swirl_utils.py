
#Code adapted from https://github.com/BRAINML-GT/SWIRL

import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

import sparse

import jax
import jax.numpy as jnp
from jax import lax, vmap, jit
from functools import partial
from jax.scipy.special import logsumexp as jax_logsumexp
import optax
 
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")


def compute_next_state_map(trans_probs, n_state, n_action):
    next_state_map = -1*np.ones((n_state, n_action), dtype=int)
    
    # Loop through trans_probs to find valid previous states for each x
    for x in range(n_state):
        for next_x in range(n_state):
            for a in range(n_action):
                if trans_probs[x, a, next_x] > 0:
                        next_state_map[x][a] = next_x
    
    return next_state_map

def compute_prev_state_map(trans_probs, n_state, n_action):
    prev_state_map = {x: [] for x in range(n_state)}
    
    # Loop through trans_probs to find valid previous states for each x
    for x in range(n_state):
        for prev_x in range(n_state):
                if np.sum(trans_probs[prev_x, :, x]) > 0:
                        prev_state_map[x].append(prev_x)
    return prev_state_map

def preprocess_xs_prev_np(xs_list, xs_prev_list, prev_state_map, n_action, n_state):
    prev_indices_list = []
    for xs, xs_prev in zip(xs_list, xs_prev_list):
        prev_indices = []
        for x, prev_x in zip(xs, xs_prev):
            try:
                prev_indices.append(prev_state_map[x].index(prev_x))
            except ValueError:
                pass
        prev_indices_list.append(prev_indices)
    return np.array(prev_indices_list)


def one_hot_jax(hidden_states, n_hidden_states):
    z = jnp.atleast_1d(hidden_states).astype(int)
    shp = hidden_states.shape
    n_samples = hidden_states.size
    zoh = jnp.zeros((n_samples, n_hidden_states))
    zoh = zoh.at[jnp.arange(n_samples), jnp.ravel(hidden_states)].set(1)
    zoh = jnp.reshape(zoh, shp + (n_hidden_states,))
    return zoh

def one_hot_jax2(hidden, hidden_prev, n_hidden_states, n_actions):
    hidden = hidden * n_actions + hidden_prev
    hidden = jnp.atleast_1d(hidden).astype(int)
    n_hidden_states_2 = n_hidden_states * n_actions
    shp = hidden.shape
    n_samples = hidden.size
    zoh = jnp.zeros((n_samples, n_hidden_states_2))
    zoh = zoh.at[jnp.arange(n_samples), jnp.ravel(hidden)].set(1)
    zoh = jnp.reshape(zoh, shp + (n_hidden_states_2,))
    return zoh

def one_hotx_partial(xs):
    global n_states
    n_states = 40 #TODO
    return one_hot_jax(xs[:, None], n_states)

def one_hotx2_partial(xs, xs_prev):
    global n_states 
    n_states = 40 #TODO
    n_actions = 9 #TODO
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states, n_actions)

def one_hota_partial(acs):
    global n_actions
    n_actions = 9 #TODO
    return one_hot_jax(acs[:, None], n_actions)


def normalize_reward(reward, indices=[0]):
    out = reward.copy()
    if isinstance(indices, int):
        indices = [indices]
    for k in indices:
        x = out[k, 0, :]
        x = (x - x.min())**4
        x = x / x.max()
        out[k, 0, :] = x
    return out


def normalize(vals):
    """
    normalize to (0, max_val)
    input:
      vals: 1d array
    """
    min_val = np.min(vals)
    max_val = np.max(vals)
    return (vals - min_val) / (max_val - min_val)


#I don't think this is a function we can leverage for our use cases - there are no "invalid" movements
#This will likely mean a significant slow down - TODO review this decision later
#def construct_new_trans_probs_limited(trans_probs, prev_state_map, next_state_map, n_state, n_action):
#    new_trans_probs = np.zeros((n_state * n_action, n_action, n_state * n_action))
#    invalid_indices = np.ones((n_state, n_action), dtype=bool)
#    for x in range(n_state):
#        for prev_x_i in np.arange(n_action):
#            if prev_x_i < len(prev_state_map[x]):
#                invalid_indices[x, prev_x_i] = False
#            new_state = x * n_action + prev_x_i
#            for a in range(n_action):
#                next_x = next_state_map[x, a]
#                new_next_state = next_x * n_action + prev_state_map[next_x].index(x)
#                new_trans_probs[new_state, a, new_next_state] = trans_probs[x, a, next_x]
#    return new_trans_probs, invalid_indices


