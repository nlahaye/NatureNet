"""
MLP reward network for NN-SWIRL.

Input: a vector over the S*S (state, previous state) pairs, pair index = s * S + p.
    S-2 uses a one-hot pair; S-1 uses the "expanded" input (1/S on every pair (s, .)).
Output: reward R[k, a] for every hidden mode k and action a.

Changes vs. the original:
- no bfloat16 cast of the input (one-hots are exact anyway; 1/S was rounded);
- the tanh was computed into an unused variable (``X = nn.tanh(x)``) so the reward was unbounded.
  ``reward_bound`` (default None = unbounded, the behaviour that was actually trained) gives
  R = reward_bound * tanh(.) when set;
- batch-safe output shape (..., K, A) (was hard-wired to a single input);
- ``head`` + ``reward_table_*``: the first layer is linear, so the reward for every pair is
  kernel[pair] + bias -> head(). This avoids building S^2 x S^2 one-hot inputs (S=75: 32M entries).
Parameter names (dense1 / dense4) are unchanged, so old saved params still load.
"""
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import optax


class MLP(nn.Module):
    hidden_size: int
    n_hidden: int
    n_actions: int
    reward_bound: float = None

    def setup(self):
        self.dense1 = nn.Dense(self.hidden_size)
        self.dense4 = nn.Dense(self.n_hidden * self.n_actions)

    def head(self, h):
        """Layers after the first (linear) one; h = pre-activation of dense1, shape (..., hidden)."""
        x = nn.leaky_relu(h)
        x = self.dense4(x)
        if self.reward_bound is not None:
            x = self.reward_bound * nn.tanh(x)
        return x.reshape(x.shape[:-1] + (self.n_hidden, self.n_actions))

    def __call__(self, x):
        return self.head(self.dense1(x))


def _first_layer(params):
    return params["dense1"]["kernel"], params["dense1"]["bias"]


def reward_table_pairs(params, apply_fn, n_states):
    """S-2 reward for every pair: (K, S, S, A), indexed [k, s, p, a]."""
    W, b = _first_layer(params)
    r = apply_fn({"params": params}, W + b, method=MLP.head)  # (S*S, K, A)
    return r.reshape(n_states, n_states, r.shape[-2], r.shape[-1]).transpose(2, 0, 1, 3)


def reward_table_expand(params, apply_fn, n_states):
    """S-1 reward on the expanded input (mean over previous states): (K, S, A)."""
    W, b = _first_layer(params)
    h = W.reshape(n_states, n_states, -1).mean(axis=1) + b  # == dense1(expanded one-hot of s)
    r = apply_fn({"params": params}, h, method=MLP.head)  # (S, K, A)
    return r.transpose(1, 0, 2)


def create_model(rng, n_states, n_hidden, hidden_size, n_actions, reward_bound=None):
    model = MLP(hidden_size=hidden_size, n_hidden=n_hidden, n_actions=n_actions, reward_bound=reward_bound)
    input_size = n_states * n_states
    params = model.init(rng, jnp.ones((1, input_size)))["params"]
    return model, params


def create_train_state(rng, learning_rate, n_states, n_hidden, hidden_size, n_actions, reward_bound=None):
    """Training state (params + Adam state). Use different rng keys for different networks."""
    model, params = create_model(rng, n_states, n_hidden, hidden_size, n_actions, reward_bound)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
