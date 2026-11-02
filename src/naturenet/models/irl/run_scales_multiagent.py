"""
Multi-agent decision-scale experiments: every (backend, scale variant) pair through the leave-one-animal-out
multi-agent harness (run_hiavi_multiagent), then one summary table.

    python -m naturenet.models.irl.run_scales_multiagent -y config/swirl/scales_multiagent.yaml
           [--backend hiql_scales swirl_scales] [--variant S1_N2_hist S3_N1_dur] [--K 2] [--skip_null]
    python -m naturenet.models.irl.plot_scales_multiagent -y config/swirl/scales_multiagent.yaml

YAML: the multi-agent YAML (data / multiagent block as hiavi_multiagent.yaml or swirl_multiagent.yaml) plus
    multiagent.scales_backends: [hiql_scales, swirl_scales]
    multiagent.scales_defaults: {switch_history: {...}, durations: {...}, hierarchy: {...}, reward_l2, ...}
    multiagent.scales_variants: list of {name, step, order, switch_history, durations, hierarchy, coupling, ...}
        (toggles: on / off / dict as in swirl_scales.yaml; "on" uses scales_defaults)
Steps (manuscript order):
    1  order N + switching history, destination coupling alpha
    2  per-animal regimes (R2), shared alpha
    3  durations with partner-driven hazards beta w_t (1 - c_t[z]) (+ alpha when K > 2)
Each pair writes <out_dir>/multiagent_<backend>_<variant>/ (ma_cv.csv, ma_summary.json, ma_final_K<K>.npz) and
the driver writes <out_dir>/ma_scales_summary.csv: pooled / per-step held-out gain (coupled - independent),
animals improved, alpha / beta medians, null p-values (p_gain, p_alpha, p_beta; NaN with --skip_null).
Joint (group-level) regimes are not modelled: regimes are per animal.
"""
import argparse
import copy
import os

import pandas as pd

from naturenet.models.irl import swirl_scales as sc
from naturenet.models.irl.run_hiavi_multiagent import expand_inputs, read_conf, run_multiagent
from naturenet.models.irl.run_swirl_scales_kfold import LATENT_KEYS, _toggle

BACKENDS = ("hiql_scales", "swirl_scales")


def scale_variants(yml_conf):
    """[(name, step, variant dict)] from multiagent.scales_variants (toggles expanded with scales_defaults)."""
    m = yml_conf.get("multiagent") or {}
    defaults = m.get("scales_defaults") or {}
    shared = {k: v for k, v in defaults.items() if k not in LATENT_KEYS}
    out = []
    for v in m.get("scales_variants") or [dict(name="N1")]:
        v = copy.deepcopy(dict(v))
        name, step = str(v.pop("name")), v.pop("step", None)
        c = {**shared, "order": int(v.pop("order", 1))}
        for k in LATENT_KEYS:
            if k in v:
                t = _toggle(v.pop(k), defaults.get(k))
                if t is not None:
                    c[k] = t
        c.update(v)
        sc.ScalesConfig.from_dict(c)                    # validate early
        out.append((name, step, c))
    return out


def pair_conf(yml_conf, backend, name, vconf):
    y = copy.deepcopy(yml_conf)
    m = y.setdefault("multiagent", {})
    m["backend"] = backend
    m["scales"] = copy.deepcopy(vconf)
    m["ma_subdir"] = f"multiagent_{backend}_{name}"
    return y


def summary_rows(summary, backend, name, step, vconf):
    rows = []
    for K, sk in summary["K"].items():
        for var, e in sk["variants"].items():
            rows.append(dict(backend=backend, scale_variant=name, step=step, model=summary["model"], K=int(K),
                             coupling_variant=var, order=int(vconf.get("order", 1)),
                             durations=bool(vconf.get("durations")), regimes=bool(vconf.get("hierarchy")),
                             history=bool(vconf.get("switch_history")),
                             pooled_gain=e["pooled_d_total"], gain_per_step=e["pooled_d_per_step"],
                             gain_context_steps=e["pooled_d_conc"], n_context_steps=e["n_conc_steps"],
                             animals_improved=e["animals_improved"], n_animals=e["n_animals"],
                             alpha_median=e["alpha_median"], beta_median=e.get("beta_median"),
                             full_data_alpha=e["full_data_alpha"], full_data_beta=e.get("full_data_beta"),
                             effective_coupling=e["effective_coupling"], p_gain=e.get("p_gain"),
                             p_alpha=e.get("p_alpha"), p_beta=e.get("p_beta"), n_null=e.get("n_null", 0),
                             sync_observed=sk["sync"]["observed"], sync_p=sk["sync"]["p"]))
    return rows


def run_all(yml_conf, backends=None, only=None, Ks=None, skip_null=False):
    yml_conf = expand_inputs(yml_conf)
    m = yml_conf.get("multiagent") or {}
    backends = list(backends or m.get("scales_backends") or BACKENDS)
    variants = scale_variants(yml_conf)
    if only:
        keep = set(only)
        variants = [v for v in variants if v[0] in keep]
    print("backends", backends, "variants", [v[0] for v in variants])
    path = os.path.join(yml_conf["out_dir"], "ma_scales_summary.csv")
    rows = []
    for backend in backends:
        for name, step, vconf in variants:
            print(f"=== {backend} / {name} (step {step}) ===")
            summ = run_multiagent(pair_conf(yml_conf, backend, name, vconf), Ks=Ks, skip_null=skip_null,
                                  backend=backend)
            rows += summary_rows(summ, backend, name, step, vconf)
            new = pd.DataFrame(rows)
            if os.path.exists(path):                     # keep rows of pairs not re-run
                old = pd.read_csv(path)
                done = set(zip(new["backend"], new["scale_variant"], new["K"]))
                old = old[[(b, v, k) not in done for b, v, k in zip(old["backend"], old["scale_variant"], old["K"])]]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(path, index=False)
    print("wrote", path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yaml", required=True)
    ap.add_argument("--backend", nargs="*", default=None, help="hiql_scales | swirl_scales")
    ap.add_argument("--variant", nargs="*", default=None)
    ap.add_argument("--K", type=int, nargs="*", default=None)
    ap.add_argument("--skip_null", action="store_true", help="no time-shift null (run first; nulls are costly)")
    a = ap.parse_args()
    df = run_all(read_conf(a.yaml), a.backend, a.variant, a.K, a.skip_null)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
