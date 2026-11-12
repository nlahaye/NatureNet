"""
Input readiness report for any SWIRL / HIAVI / multi-agent YAML (no fitting).

    python -m naturenet.models.irl.check_inputs -y <config.yaml> [--out <csv>]

Loads the sequences exactly as the runners do (actions / positions / seq_len / segment_policy, hiavi_align
swirl) and the step times from <df_dir>/<df_run_uid>_dfs.pkl, then reports:
  * states / actions in range for n_states (the loaders raise on the legacy states format and on bad codes);
  * state usage: states never visited, share of steps in the 5 most visited states;
  * per animal (original tracks only): sequences, steps, calendar span, step interval dt (median, p10, p90)
    and the share of steps with dt > 2x the dataset median;
  * env window coverage (multiagent.window, if set): animals and steps inside the window;
  * flags: irregular dt (durations / hazards are counted in steps), whether each animal has enough steps for
    seq_len, and whether the window leaves >= 2 animals with overlap.
"""
import argparse
import os

import numpy as np
import pandas as pd

from naturenet.models.irl.run_hiavi_kfold import prepare_hiavi_data
from naturenet.models.irl.run_hiavi_multiagent import (_to_days, original_mask, read_conf, resolve_window,
                                                       step_metadata)


def report(conf):
    data = prepare_hiavi_data(conf)
    S = int(conf["n_states"])
    s = np.asarray(data["hiavi_s"])
    pc = step_metadata(conf, data)
    orig = original_mask(pc)
    N, T = s.shape
    valid = pc.valid                                                           # real steps (variable lengths)
    n_real = valid.sum(1)
    dt_h = np.diff(pc.times, axis=1) * 24.0                                    # (N, T-1) hours; NaN at padding
    med = float(np.nanmedian(dt_h[orig])) if orig.any() else float(np.nanmedian(dt_h))
    counts = np.bincount(s[orig[:, None] & valid], minlength=S)
    top5 = float(np.sort(counts)[::-1][:5].sum() / max(counts.sum(), 1))
    w = resolve_window((conf.get("multiagent") or {}).get("window"))
    lo, hi = (float(_to_days([w[0]])[0]), float(_to_days([w[1]])[0]) + 1.0) if w else (-np.inf, np.inf)
    rows = []
    for an in sorted(set(pc.animal[orig])):
        m = orig & (pc.animal == an)
        d = dt_h[m].ravel()
        d = d[np.isfinite(d)]
        t = pc.times[m]
        t = t[np.isfinite(t)]
        rows.append(dict(animal=an, n_seq=int(m.sum()), n_steps=int(n_real[m].sum()),
                         start=str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(t.min())))[:10],
                         end=str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=float(t.max())))[:10],
                         dt_med_h=float(np.median(d)), dt_p10_h=float(np.percentile(d, 10)),
                         dt_p90_h=float(np.percentile(d, 90)), frac_dt_gt_2med=float(np.mean(d > 2 * med)),
                         steps_in_window=int(((t >= lo) & (t < hi)).sum())))
    per = pd.DataFrame(rows)
    dto = dt_h[orig]
    dto = dto[np.isfinite(dto)]
    summ = dict(n_sequences=N, seq_len_steps=T if (n_real == T).all() else
                f"{int(n_real.min())}-{T} (median {np.median(n_real):.0f}, variable)",
                n_original_sequences=int(orig.sum()),
                n_animals=len(per), n_states=S, states_unvisited=int((counts == 0).sum()),
                top5_state_share=top5, dt_median_h=med,
                frac_dt_gt_2med=float(np.mean(dto > 2 * med)) if dto.size else 0.0,
                animals_in_window=int((per["steps_in_window"] > 0).sum()) if w else None)
    flags = []
    if summ["frac_dt_gt_2med"] > 0.2:
        flags.append(f"irregular dt: {100 * summ['frac_dt_gt_2med']:.0f}% of steps exceed 2x the median "
                     f"({med:.1f} h). Durations / hazards count steps; report them in steps, or rerun step-3 "
                     "variants on regularised tracks as a sensitivity.")
    if summ["states_unvisited"] > 0.3 * S:
        flags.append(f"{summ['states_unvisited']} of {S} states never visited by the original tracks; "
                     "check n_states and the env date coverage.")
    if w and summ["animals_in_window"] < 2:
        flags.append("fewer than 2 animals inside multiagent.window: no partner context.")
    thin = per[per["n_seq"] < 2]["animal"].tolist()
    if thin:
        flags.append(f"animals with < 2 sequences at seq_len {summ['seq_len_steps']}: {thin} (weak held-out folds)")
    return summ, per, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--out", default=None, help="per-animal CSV")
    a = ap.parse_args()
    summ, per, flags = report(read_conf(a.yaml))
    for k, v in summ.items():
        print(f"{k:24s} {v:.3f}" if isinstance(v, float) else f"{k:24s} {v}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(per.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("FLAGS:" if flags else "FLAGS: none")
    for f in flags:
        print(" -", f)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        per.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
