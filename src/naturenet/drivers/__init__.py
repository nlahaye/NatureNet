"""
Environmental-driver tests and representation-geometry analyses for NatureNet IRL runs (v2 environments).

Generic: every tool reads the same run YAMLs as the fitting code (run_hiavi_kfold / run_swirl_scales_kfold /
run_hiavi_multiagent) plus a small block of its own, so the procedures carry over to other SIT-FUSE use cases
(any state space built from SIT-FUSE clusters, any species / tracking data set).

    env_null              Test 1  environment ablation nulls (shift / shuffle / marginal / single) as state "levels"
    env_scale_grid        Test 2  environment scale x decision scale grid (paired fold / animal gains)
    year_folds            Test 3  leave-year-out folds (whales 2015 <-> 2017; sharks by year)
    ppc                   Test 4  posterior predictive simulation + one-step predictive calibration
    driver_consistency    Test 5  reward -> environment attribution at every level / model / fold, agreement
    coupling_env          Test 6  environment-matched social null (summary; the null lives in run_hiavi_multiagent)
    env_covariates        Test 7  environmental covariates for the intention switches (swirl_scales history "env")
    switch_env                    switch-triggered environmental averages of decoded intentions
    availability                  available points (step-selection style) for use-availability geometry
    env_geometry                  intrinsic dimension, LID spatial structure, neighbourhood overlap, level purity
    projection_robustness         level silhouette across native / PCA / UMAP / t-SNE projections
    synthesis                     cross-perspective synthesis: rank agreement, Friedman / Nemenyi, variance shares
    reptax_bridge                 Rep_Taxonomy ports (ID, Moran's I, Mantel, Kendall W, Friedman / Nemenyi, SS shares)
    resources                     job wall time / CPU / GPU hours from the launcher state dirs (+ peak RSS wrapper)
    audit                         read-only experiment auditor (done / missing / stale / failed per unit)
    plan                          sit_fuse.local_launcher plan for all of the above (parallel, resumable)
"""
