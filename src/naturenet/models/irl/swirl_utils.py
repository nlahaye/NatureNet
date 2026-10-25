
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
# Platform is left to JAX (uses the GPU when present, else CPU).
# jax.config.update("jax_platform_name", "gpu")


from naturenet.environment.pipeline_checks import check_s2_prev_slots

DEFAULT_S2_MAX_DENSE_ENTRIES = int(2e8)


def _support_coords(trans_probs):
    """Nonzero (x, a, x') of a dense or sparse.COO transition tensor."""
    if hasattr(trans_probs, "coords"):
        c = np.asarray(trans_probs.coords)
        d = np.asarray(trans_probs.data)
        keep = d > 0
        return c[0][keep], c[1][keep], c[2][keep], d[keep]
    T = np.asarray(trans_probs)
    x, a, nx = np.nonzero(T > 0)
    return x, a, nx, T[x, a, nx]


def compute_next_state_map(trans_probs, n_state, n_action):
    """
    Most likely successor per (x, a); -1 if (x, a) has no outgoing mass. Vectorised.
    (The old triple loop kept the LAST nonzero successor, which is arbitrary for stochastic T.)
    Only used for diagnostics now; construct_new_trans_probs_limited uses the full T.
    """
    T = trans_probs.todense() if hasattr(trans_probs, "todense") else np.asarray(trans_probs)
    nxt = np.argmax(T, axis=2).astype(int)
    nxt[T.sum(axis=2) <= 0] = -1
    return nxt


def compute_prev_state_map(trans_probs, n_state, n_action, observed_pairs=None):
    """
    x -> sorted list of predecessors prev_x with sum_a T[prev_x, a, x] > 0, plus any
    (prev_x, x) pairs observed in the data, given as [(xs, xs_prev), ...], so every observed
    pair has a slot. Vectorised.
    """
    x, _, nx, _ = _support_coords(trans_probs)
    pairs = set(zip(nx.tolist(), x.tolist()))  # (x, prev_x)
    if observed_pairs is not None:
        for xs, xs_prev in observed_pairs:
            pairs.update(zip(np.asarray(xs).ravel().tolist(), np.asarray(xs_prev).ravel().tolist()))
    prev_state_map = {s: [] for s in range(n_state)}
    for cur, prev in pairs:
        prev_state_map[int(cur)].append(int(prev))
    for s in prev_state_map:
        prev_state_map[s].sort()
    return prev_state_map


def n_prev_slots(prev_state_map):
    """M = max number of predecessors of any state (was hard-wired to n_actions = 9)."""
    return max(1, max((len(v) for v in prev_state_map.values()), default=1))


def prev_slot_lookup(prev_state_map):
    return {(x, p): j for x, prevs in prev_state_map.items() for j, p in enumerate(prevs)}


def preprocess_xs_prev_np(xs_list, xs_prev_list, prev_state_map, n_action, n_state):
    """
    Slot index of prev_x within prev_state_map[x] for every step. Raises on a missing pair
    (the old version silently skipped it, producing ragged / misaligned sequences).
    """
    lut = prev_slot_lookup(prev_state_map)
    xs_arr = np.asarray(xs_list)
    xp_arr = np.asarray(xs_prev_list)
    if xs_arr.shape != xp_arr.shape:
        raise ValueError(f"xs {xs_arr.shape} and xs_prev {xp_arr.shape} must have the same shape")
    out = np.empty(xs_arr.shape, dtype=np.int64)
    flat_x, flat_p, flat_o = xs_arr.ravel(), xp_arr.ravel(), out.reshape(-1)
    for i in range(flat_x.size):
        j = lut.get((int(flat_x[i]), int(flat_p[i])))
        if j is None:
            raise ValueError(f"(prev={int(flat_p[i])} -> x={int(flat_x[i])}) has no predecessor slot. Build "
                             f"prev_state_map with observed_pairs that include this data.")
        flat_o[i] = j
    return out


def construct_new_trans_probs_limited(trans_probs, prev_state_map, next_state_map=None, n_state=None, n_action=None,
                                      n_slots=None, max_dense_entries=DEFAULT_S2_MAX_DENSE_ENTRIES):
    """
    S-2 (second-order) transition tensor over augmented states (x, j), j = slot of the previous state.

        new_T[(x, j), a, (x', slot_of(x', x))] = T[x, a, x']    for every x' in the support of T[x, a]

    Stochastic T is handled (the old version used one successor per (x, a)). Slots j >= len(prev[x])
    are unused (flagged in invalid_indices) but get x's transitions so their rows still sum to 1.
    ``next_state_map`` is accepted for backward compatibility and ignored.
    Returns (new_T dense (S*M, A, S*M), invalid_indices (S, M)).
    """
    if n_state is None or n_action is None:
        n_state, n_action = trans_probs.shape[0], trans_probs.shape[1]
    M = n_slots if n_slots is not None else n_prev_slots(prev_state_map)
    check_s2_prev_slots(M, n_state, n_action, max_dense_entries)

    lut = prev_slot_lookup(prev_state_map)
    x, a, nx, p = _support_coords(trans_probs)
    tgt_slot = np.array([lut.get((int(n), int(c)), -1) for n, c in zip(nx, x)], dtype=np.int64)
    if np.any(tgt_slot < 0):
        raise ValueError("prev_state_map is missing predecessors present in trans_probs; rebuild it from this T")

    new_T = np.zeros((n_state * M, n_action, n_state * M))
    cols = nx * M + tgt_slot
    for j in range(M):
        new_T[x * M + j, a, cols] = p

    invalid_indices = np.ones((n_state, M), dtype=bool)
    for s, prevs in prev_state_map.items():
        invalid_indices[s, :len(prevs)] = False
    return new_T, invalid_indices


def one_hot_jax_nn(hidden, n_hidden_states):
    hidden = jnp.atleast_1d(hidden).astype(int)
    shp = hidden.shape
    n_samples = hidden.size
    zoh = jnp.zeros((n_samples, n_hidden_states))
    zoh = zoh.at[jnp.arange(n_samples), jnp.ravel(hidden)].set(1)
    zoh = jnp.reshape(zoh, shp + (n_hidden_states,))
    return zoh

def one_hot_jax2_nn(hidden, hidden_prev, n_hidden_states):
    hidden = hidden * n_hidden_states + hidden_prev
    hidden = jnp.atleast_1d(hidden).astype(int)
    n_hidden_states_2 = n_hidden_states * n_hidden_states
    shp = hidden.shape
    n_samples = hidden.size
    zoh = jnp.zeros((n_samples, n_hidden_states_2))
    zoh = zoh.at[jnp.arange(n_samples), jnp.ravel(hidden)].set(1)
    zoh = jnp.reshape(zoh, shp + (n_hidden_states_2,))
    return zoh, hidden


def one_hot_jax(hidden_states, n_hidden_states):
    hidden = jnp.atleast_1d(hidden_states).astype(int)
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

def one_hotx_partial(xs, n_states):
    #n_states = 98 #146 #TODO
    return one_hot_jax(xs[:, None], n_states)

def one_hotx2_partial(xs, xs_prev, n_states, n_actions):
    # NOTE: n_actions here is the number of predecessor SLOTS M (see n_prev_slots); it was 9 when
    # S-2 assumed at most 9 grid neighbours. Pass M for empirical / abstract-state transitions.
    #n_states = 98 # 146TODO
    #n_actions = 9 #TODO
    return one_hot_jax2(xs[:, None], xs_prev[:, None], n_states, n_actions)

def one_hota_partial(acs, n_actions):
    #n_actions = 9 #TODO
    return one_hot_jax(acs[:, None], n_actions)

def one_hotx_partial_nn(xs, n_states):
    #n_states = 98 #146 #TODO
    return one_hot_jax_nn(xs[:, None], n_states)

def one_hotx2_partial_nn(xs, xs_prev, n_states):
    #n_states = 98 # 146TODO
    return one_hot_jax2_nn(xs[:, None], xs_prev[:, None], n_states)

def one_hota_partial_nn(acs, n_actions):
    #n_actions = 9 #TODO
    return one_hot_jax_nn(acs[:, None], n_actions)


def normalize_reward(reward, indices=[0]):
    """For PLOTTING only: (r - min)^4 / max is a nonlinear transform; never feed it back into LL computations."""
    out = np.array(reward, dtype=np.float64, copy=True)
    if isinstance(indices, int):
        indices = [indices]
    for k in indices:
        x = out[k, 0, :]
        x = (x - x.min())**4
        mx = x.max()
        x = x / mx if mx > 0 else x
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
    rng = max_val - min_val
    return (vals - min_val) / rng if rng > 0 else np.zeros_like(vals)
