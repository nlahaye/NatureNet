"""
Environmental covariates aligned with the NatureNet sequences (test 7 and the geometry tools).

The raw channels at every fix come from naturenet.environment.sample_env_inputs:
  <df_dir>/<df_run_uid>_<key>_states_input.pkl    list (per segment) of float32 (len(segment), C), NaN = missing
  <df_dir>/<df_run_uid>_states_input_channels.json channel names
Sequence row position p of sequence i (data["states"][i, p]) is fix start_i + p of its segment, so

  env_rows(conf, data, channels)   (N, L, C) aligned with data["states"] (for swirl_scales switch_history "env")
  env_steps(conf, data, channels)  (N, T, C) aligned with the HIAVI steps (hiavi_s; row position p = t + offset)

Channels are z-scored over all real fixes of the run (covariate standardisation only; no response is used).
Optional per-channel transforms: env_transforms: {chl: log10, eke: log1p}.

YAML use (decision-scale variant with environment-driven switching):
  scales:
    variants:
      - {switch_history: {kind: "env", env_channels: [sst, grad_mag_sst, log10_CHL, adt, eke, depth_m], env_tag: core}}
      - {switch_history: {kind: "lags+env", env_channels: [...], env_tag: core}, durations: "on"}
"""
import json
import os
import pickle

import numpy as np

TRANSFORMS = {"log10": lambda x: np.log10(np.where(x > 0, x, np.nan)), "log1p": lambda x: np.log1p(np.maximum(x, 0)),
              "abs": np.abs, "none": lambda x: x}


def channel_names(conf):
    p = conf.get("env_channels_json") or os.path.join(conf["df_dir"], conf["df_run_uid"] + "_states_input_channels.json")
    with open(p) as f:
        return list(json.load(f))


def _input_path(conf, key):
    from naturenet.models.irl.swirl_track_series import _df_key
    return os.path.join(conf["df_dir"], f"{conf['df_run_uid']}_{_df_key(key, conf['df_run_uid'])}_states_input.pkl")


def _load_inputs(conf, keys):
    out = {}
    for k in sorted(set(keys)):
        p = _input_path(conf, k)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} (run naturenet.environment.sample_env_inputs for this run first)")
        with open(p, "rb") as f:
            out[k] = [np.asarray(x, dtype=np.float32) for x in pickle.load(f)]
    return out


def _select(names, channels):
    if not channels:
        return list(range(len(names))), list(names)
    bad = [c for c in channels if c not in names]
    if bad:
        raise ValueError(f"env channels {bad} not in the states_input channels {names}")
    return [names.index(c) for c in channels], list(channels)


def env_rows(conf, data, channels=(), offset=0, length=None, standardize=True):
    """(N, L, C) covariates at row positions offset .. offset + L - 1 of every sequence (NaN = missing / padded)."""
    names = channel_names(conf)
    cols, sel = _select(names, channels)
    meta = data["meta"]
    L = int(length if length is not None else np.asarray(data["states"]).shape[1])
    inp = _load_inputs(conf, [m["key"] for m in meta])
    E = np.full((len(meta), L, len(cols)), np.nan, dtype=np.float64)
    for i, m in enumerate(meta):
        seg = inp[m["key"]][int(m["seg"])]
        pos = int(m["start"]) + offset + np.arange(L)
        ok = pos < len(seg)                         # padded positions are masked downstream
        E[i, ok] = seg[pos[ok]][:, cols]
    tr = conf.get("env_transforms") or {}
    for j, c in enumerate(sel):
        if c in tr:
            E[..., j] = TRANSFORMS[tr[c]](E[..., j])
    if standardize:
        mu = np.nanmean(E.reshape(-1, E.shape[-1]), 0)
        sd = np.nanstd(E.reshape(-1, E.shape[-1]), 0)
        E = (E - mu) / np.where(sd > 0, sd, 1.0)
    return E, sel


def cached_env_rows(conf, data, channels=()):
    cache = data.setdefault("_cache", {})
    key = ("env_rows", tuple(channels or ()))
    if key not in cache:
        cache[key] = env_rows(conf, data, channels)
    return cache[key]


def env_steps(conf, data, channels=(), standardize=True):
    """(N, T, C) covariates at the HIAVI steps (data from prepare_hiavi_data); NaN on padded steps."""
    s = np.asarray(data["hiavi_s"])
    E, sel = env_rows(conf, data, channels, offset=int(data["hiavi_offset"]), length=s.shape[1],
                      standardize=standardize)
    mask = np.asarray(data.get("hiavi_mask", np.ones(s.shape, bool)))
    E[~mask] = np.nan
    return E, sel
