"""
CSE516 M3 experiment runner
Extends the M2 implementation in Code.py. No results are fabricated:
all tables and figures are generated from the supplied dataset at runtime.

Run from the directory containing Code.py and this file:
    python m3_experiments.py --data "data/survey lung cancer.csv"

Optional:
    python m3_experiments.py --data "data/survey lung cancer.csv" --repeats 5 --folds 5 --bootstrap 100

Outputs are written to results_m3/.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, roc_auc_score
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import Code as m2

warnings.filterwarnings("default")

TARGET = m2.TARGET
SEED = 516
MODELS = ("Logistic Regression", "Neural Network (MLP)", "SVM (RBF)",
          "PC-learned Bayesian Network")
METRICS = ("AUROC", "AUPRC", "F1_NO", "Brier_score", "Log_loss", "ECE", "Accuracy")


def probability_metrics(y_true, prob):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-7, 1 - 1e-7)
    pred = (prob >= 0.5).astype(int)
    # Fixed-width bins make ECE definition comparable across models.
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(float(prob[mask].mean()) -
                                     float(y_true[mask].mean()))
    return {
        "AUROC": float(roc_auc_score(y_true, prob)),
        "AUPRC": float(average_precision_score(y_true, prob)),
        "F1_NO": float(f1_score(y_true, pred, pos_label=0, zero_division=0)),
        "Brier_score": float(brier_score_loss(y_true, prob)),
        "Log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "ECE": float(ece),
        "Accuracy": float(accuracy_score(y_true, pred)),
    }


def ci_mean(values, metric=None, confidence=0.95):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    sem = x.std(ddof=1) / math.sqrt(len(x))
    half = float(student_t.ppf((1 + confidence) / 2, len(x) - 1) * sem)
    lower, upper = float(x.mean() - half), float(x.mean() + half)
    if metric in {"AUROC", "AUPRC", "F1_NO", "Brier_score", "ECE", "Accuracy"}:
        lower, upper = max(0.0, lower), min(1.0, upper)
    if metric in {"AUROC", "AUPRC", "F1_NO", "ECE", "Accuracy"}:
        lower, upper = max(0.0, lower), min(1.0, upper)
    return [lower, upper]


def build_models(seed):
    binary = None
    pre = ColumnTransformer([
        ("age", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), ["AGE"]),
        ("binary", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent"))
        ]), binary if binary is not None else [])
    ], remainder="passthrough")
    # The binary columns are passed through unchanged; all current binary inputs
    # have already been validated and encoded by the M2 loader.
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=seed),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=0.01, early_stopping=True,
            max_iter=1200, random_state=seed),
        "SVM (RBF)": SVC(
            C=1.0, kernel="rbf", class_weight="balanced",
            probability=True, random_state=seed),
    }
    return pre, models


def make_preprocessor(X):
    binary = [c for c in X.columns if c != "AGE"]
    return ColumnTransformer([
        ("age", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), ["AGE"]),
        ("binary", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent"))
        ]), binary)
    ])


def repeated_predictive_experiments(df, names, card, folds, repeats, seed,
                                    alpha_pc, max_cond, protected):
    X = df.drop(columns=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    splitter = RepeatedStratifiedKFold(
        n_splits=folds, n_repeats=repeats, random_state=seed)

    rows, prediction_rows = [], []
    for split_no, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_no = split_no // folds
        fold_no = split_no % folds
        run_seed = seed + repeat_no
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pre = make_preprocessor(X_train)
        for model_name, model in build_models(run_seed)[1].items():
            pipeline = Pipeline([("preprocess", clone(pre)), ("model", clone(model))])
            pipeline.fit(X_train, y_train)
            prob = pipeline.predict_proba(X_test)[:, 1]
            vals = probability_metrics(y_test, prob)
            rows.append({"repeat": repeat_no + 1, "fold": fold_no + 1,
                         "model": model_name, **vals})
            for idx, truth, p in zip(test_idx, y_test, prob):
                prediction_rows.append({
                    "repeat": repeat_no + 1, "fold": fold_no + 1,
                    "row_index": int(idx), "model": model_name,
                    "y_true": int(truth), "probability": float(p)
                })

        # Structure and CPTs are learned only from the training fold.
        train_df = pd.concat(
            [X_train.reset_index(drop=True),
             y_train.reset_index(drop=True).rename(TARGET)], axis=1)[names]
        T = len(names) - 1
        test_int = None
        for bn_name, edge_constraints in (
            ("PC-learned Bayesian Network (protected edges)", protected),
            ("PC-learned Bayesian Network (plain PC)", set())
        ):
            bn, _, age_cuts, *_ = m2.learn_bn(
                train_df, names, card, alpha_pc, max_cond,
                protected=edge_constraints)
            test_disc, _ = m2.discretize_age(X_test, cuts=age_cuts)
            test_int = test_disc[names[:-1]].to_numpy(dtype=np.int64)
            full = np.column_stack([test_int, np.zeros(len(test_int), dtype=np.int64)])
            prob = np.asarray([bn.p_target(row, range(T), T) for row in full])
            vals = probability_metrics(y_test, prob)
            rows.append({"repeat": repeat_no + 1, "fold": fold_no + 1,
                         "model": bn_name, **vals})
            for idx, truth, p in zip(test_idx, y_test, prob):
                prediction_rows.append({
                    "repeat": repeat_no + 1, "fold": fold_no + 1,
                    "row_index": int(idx), "model": bn_name,
                    "y_true": int(truth), "probability": float(p)
                })

        print(f"Predictive CV: repeat {repeat_no + 1}/{repeats}, "
              f"fold {fold_no + 1}/{folds}", flush=True)

    fold_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(prediction_rows)

    # Pool the out-of-fold predictions within each repeat. These are the primary
    # repeated-run observations used for mean +/- SD and paired comparisons.
    repeat_rows = []
    for (rep, model_name), group in pred_df.groupby(["repeat", "model"]):
        # Each observation appears once per repeat in the OOF prediction set.
        vals = probability_metrics(group["y_true"], group["probability"])
        repeat_rows.append({"repeat": int(rep), "model": model_name, **vals})
    repeat_df = pd.DataFrame(repeat_rows)
    return fold_df, repeat_df, pred_df


def summarize_repeats(repeat_df):
    out = []
    for model_name, group in repeat_df.groupby("model", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            lo, hi = ci_mean(values, metric=metric)
            out.append({
                "model": model_name, "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95_lower": lo, "ci95_upper": hi,
                "repeats": int(len(values))
            })
    return pd.DataFrame(out)


def paired_statistics(repeat_df, reference="PC-learned Bayesian Network (plain PC)"):
    rows = []
    for metric in METRICS:
        wide = repeat_df.pivot(index="repeat", columns="model", values=metric)
        if reference not in wide:
            continue
        for model_name in wide.columns:
            if model_name == reference:
                continue
            pair = wide[[model_name, reference]].dropna()
            a, b = pair[model_name].to_numpy(), pair[reference].to_numpy()
            if len(a) < 2:
                continue
            try:
                t_p = float(ttest_rel(a, b).pvalue)
            except Exception:
                t_p = float("nan")
            try:
                w_p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except ValueError:
                w_p = 1.0
            rows.append({
                "metric": metric, "comparison": f"{model_name} vs {reference}",
                "n_paired_repeats": len(a),
                "mean_difference_first_minus_reference": float(np.mean(a - b)),
                "paired_t_test_p": t_p,
                "wilcoxon_p": w_p,
                "interpretation_note":
                    "Exploratory: repeated-CV runs are not fully independent; "
                    "do not treat p-values as definitive."
            })
    return pd.DataFrame(rows)


def structure_sensitivity(df, alpha_values, max_cond, protected_edges, out):
    rows, edge_rows = [], []
    names = list(df.columns)
    for alpha in alpha_values:
        for mode, protected in (
            ("Plain PC", set()),
            ("Protected-edge PC", protected_edges)
        ):
            disc, cuts = m2.discretize_age(df)
            graph, seps, log = m2.pc_skeleton(
                disc, alpha=alpha, max_cond=max_cond, protected=protected)
            rows.append({
                "alpha": alpha, "method": mode,
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "ci_tests": len(log),
                "separating_sets": len(seps),
                "max_conditioning_level": max(
                    [x["level"] for x in log], default=0)
            })
            for u, v in graph.edges():
                edge_rows.append({
                    "alpha": alpha, "method": mode,
                    "edge": " -- ".join(sorted((u, v)))
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "pc_alpha_sensitivity.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "pc_edges_by_alpha.csv", index=False)
    return summary


def bootstrap_edge_stability(df, alpha, max_cond, B, seed, protected, out):
    names = list(df.columns)
    possible = [(names[i], names[j]) for i in range(len(names))
                for j in range(i + 1, len(names))]
    counts = {tuple(sorted(e)): 0 for e in possible}
    rng = np.random.default_rng(seed)
    for b in range(B):
        sample = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        disc, _ = m2.discretize_age(sample)
        graph, _, _ = m2.pc_skeleton(
            disc, alpha=alpha, max_cond=max_cond, protected=protected)
        for u, v in graph.edges():
            counts[tuple(sorted((u, v)))] += 1
        print(f"Bootstrap structure: {b + 1}/{B}", flush=True)
    result = pd.DataFrame([
        {"u": e[0], "v": e[1], "selection_frequency": c / B,
         "bootstrap_runs": B}
        for e, c in counts.items()
    ]).sort_values("selection_frequency", ascending=False)
    result.to_csv(out / "pc_bootstrap_edge_stability.csv", index=False)
    return result


def sample_size_experiment(df, fractions, repeats, alpha, max_cond, seed, out):
    rng = np.random.default_rng(seed)
    rows = []
    for fraction in fractions:
        n = max(20, int(round(len(df) * fraction)))
        for rep in range(repeats):
            idx = rng.choice(len(df), size=n, replace=False)
            sample = df.iloc[idx].reset_index(drop=True)
            disc, _ = m2.discretize_age(sample)
            graph, _, log = m2.pc_skeleton(
                disc, alpha=alpha, max_cond=max_cond, protected=set())
            rows.append({
                "fraction": fraction, "sample_size": n, "repeat": rep + 1,
                "edges": graph.number_of_edges(), "density": nx.density(graph),
                "ci_tests": len(log)
            })
        print(f"Sample-size experiment: {fraction:.0%} complete", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "sample_size_structure_robustness.csv", index=False)
    return result


def perturb_test_set(X, mode, rate, rng, train_X):
    """Apply test-time MCAR missingness or binary/continuous feature noise."""
    corrupted = X.copy()
    feature_cols = list(corrupted.columns)
    if mode == "mcar":
        mask = rng.random(corrupted.shape) < rate
        corrupted = corrupted.mask(mask)
    elif mode == "noise":
        for col in feature_cols:
            if col == "AGE":
                scale = max(float(train_X[col].std()), 1.0)
                corrupted[col] = corrupted[col] + rng.normal(
                    0, rate * scale, size=len(corrupted))
            else:
                flip = rng.random(len(corrupted)) < rate
                corrupted.loc[flip, col] = 1 - corrupted.loc[flip, col]
    return corrupted


def robustness_prediction_experiments(df, names, card, folds, seed,
                                      alpha_pc, max_cond, protected, out):
    """Train on clean folds; corrupt only held-out features to isolate robustness."""
    X = df.drop(columns=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=1, random_state=seed)
    settings = [("clean", 0.0), ("mcar_10pct", 0.10), ("mcar_20pct", 0.20),
                ("noise_05", 0.05), ("noise_10", 0.10)]
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
        ytr, yte = y.iloc[tr], y.iloc[te]
        pre = make_preprocessor(Xtr)
        fitted = {}
        for model_name, model in build_models(seed + fold)[1].items():
            pipe = Pipeline([("preprocess", clone(pre)), ("model", clone(model))])
            pipe.fit(Xtr, ytr)
            fitted[model_name] = pipe

        train_df = pd.concat(
            [Xtr.reset_index(drop=True), ytr.reset_index(drop=True).rename(TARGET)],
            axis=1)[names]
        bn, _, cuts, *_ = m2.learn_bn(
            train_df, names, card, alpha_pc, max_cond, protected=protected)

        for setting, rate in settings:
            rng = np.random.default_rng(seed + fold * 100 + int(rate * 1000) +
                                        (0 if setting == "clean" else len(setting)))
            Xcorrupt = Xte.copy() if setting == "clean" else perturb_test_set(
                Xte, "mcar" if setting.startswith("mcar") else "noise",
                rate, rng, Xtr)
            for model_name, pipe in fitted.items():
                prob = pipe.predict_proba(Xcorrupt)[:, 1]
                rows.append({"fold": fold, "condition": setting,
                             "model": model_name,
                             **probability_metrics(yte, prob)})

            # BN cannot consume missing values; use train-fold modes/median for
            # missing values, then apply the training-fold AGE cut points.
            Xbn = Xcorrupt.copy()
            for col in Xbn.columns:
                if Xbn[col].isna().any():
                    fill = (float(Xtr[col].median()) if col == "AGE"
                            else float(Xtr[col].mode().iloc[0]))
                    Xbn[col] = Xbn[col].fillna(fill)
            test_disc, _ = m2.discretize_age(Xbn, cuts=cuts)
            T = len(names) - 1
            arr = test_disc[names[:-1]].to_numpy(dtype=np.int64)
            full = np.column_stack([arr, np.zeros(len(arr), dtype=np.int64)])
            prob = np.asarray([bn.p_target(r, range(T), T) for r in full])
            rows.append({"fold": fold, "condition": setting,
                         "model": "PC-learned Bayesian Network",
                         **probability_metrics(yte, prob)})
        print(f"Robustness CV: fold {fold}/{folds}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "robustness_metrics.csv", index=False)
    return result


def graphical_lasso_comparison(df, names, out):
    partial_corr, selected_alpha = m2.learn_ci_graph(df, names)
    rows = []
    graph = nx.Graph()
    graph.add_nodes_from(names)
    for i, u in enumerate(names):
        for j in range(i + 1, len(names)):
            value = float(partial_corr[i, j])
            if value != 0.0:
                v = names[j]
                graph.add_edge(u, v, partial_correlation=value)
                rows.append({"u": u, "v": v, "partial_correlation": value,
                             "absolute_partial_correlation": abs(value)})
    pd.DataFrame(rows).to_csv(out / "graphical_lasso_edges.csv", index=False)
    summary = {
        "method": "Graphical Lasso",
        "selected_alpha": selected_alpha,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "interpretation": "Undirected conditional-dependence estimate; not a causal graph."
    }
    (out / "graphical_lasso_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def inference_experiments(df, names, card, alpha, max_cond, protected, out, seed):
    disc, _ = m2.discretize_age(df)
    skel, seps, log = m2.pc_skeleton(
        disc, alpha=alpha, max_cond=max_cond, protected=protected)
    cpdag, dag = m2.orient_cpdag(skel, seps)
    bn = m2.DiscreteBN(names, card, dag, alpha=1.0).fit(
        m2.to_int(disc, names))
    X = m2.to_int(disc, names)
    target_idx = names.index(TARGET)
    rng = np.random.default_rng(seed)
    rows = []
    for n_evidence in range(0, target_idx + 1):
        times = []
        errors = []
        for _ in range(30):
            observed = rng.choice(target_idx, size=n_evidence, replace=False)
            row = X[rng.integers(len(X))]
            evidence = {int(j): int(row[j]) for j in observed}
            start = time.perf_counter()
            posterior = bn.posterior(target_idx, evidence)
            times.append(time.perf_counter() - start)
            if n_evidence <= 4:
                brute = bn.brute_force_posterior(target_idx, evidence)
                errors.append(float(np.max(np.abs(posterior - brute))))
        rows.append({
            "number_of_observed_variables": n_evidence,
            "mean_inference_time_ms": float(np.mean(times) * 1000),
            "std_inference_time_ms": float(np.std(times, ddof=1) * 1000),
            "max_ve_vs_bruteforce_error": max(errors) if errors else None,
            "queries": len(times)
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "inference_runtime_validation.csv", index=False)
    return result, cpdag, dag, skel, log


def error_analysis(pred_df, out):
    rows = []
    for model_name, group in pred_df.groupby("model"):
        y = group["y_true"].to_numpy(dtype=int)
        p = group["probability"].to_numpy(dtype=float)
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "model": model_name, "threshold": 0.5,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "specificity_NO_recall": float(tn / (tn + fp)) if tn + fp else 0.0,
            "sensitivity_YES_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
            "note": "Counts pool repeated out-of-fold predictions; the same row appears once per repeat."
        })
    result = pd.DataFrame(rows)
    result.to_csv(out / "error_analysis_confusion_matrices.csv", index=False)
    return result


def calibration_plot(pred_df, out):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, group in pred_df.groupby("model"):
        y = group["y_true"].to_numpy()
        p = group["probability"].to_numpy()
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
            if mask.sum():
                xs.append(float(p[mask].mean()))
                ys.append(float(y[mask].mean()))
        ax.plot(xs, ys, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability",
           ylabel="Observed positive fraction",
           title="Reliability diagram (out-of-fold predictions)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability_diagram.png", dpi=220)
    plt.close(fig)


def plot_results(summary, structure, bootstrap, sample_size, robustness,
                 inference, out):
    # Mean KPI comparison, separated by metric scale.
    for metric in ("AUROC", "AUPRC", "F1_NO", "Brier_score", "Log_loss", "ECE"):
        sub = summary[summary.metric == metric]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(sub.model, sub["mean"], yerr=sub["std"], capsize=4)
        ax.set_ylabel(metric + (" (lower is better)" if metric in
                               ("Brier_score", "Log_loss", "ECE") else
                               " (higher is better)"))
        ax.set_title(f"{metric}: repeated-CV mean +/- SD")
        ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        fig.savefig(out / f"kpi_{metric.lower()}.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in structure.groupby("method"):
        ax.plot(group.alpha, group.edges, marker="o", label=method)
    ax.set(xlabel="PC significance level (alpha)", ylabel="Skeleton edge count",
           title="PC structure sensitivity to alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pc_alpha_edge_sensitivity.png", dpi=220)
    plt.close(fig)

    if not bootstrap.empty:
        top = bootstrap.head(15).sort_values("selection_frequency")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["u"] + " -- " + top["v"], top.selection_frequency)
        ax.set(xlabel="Bootstrap selection frequency",
               title="Top 15 PC edge stability frequencies")
        fig.tight_layout()
        fig.savefig(out / "pc_edge_stability.png", dpi=220)
        plt.close(fig)

    if not sample_size.empty:
        agg = sample_size.groupby("fraction").edges.agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(agg.fraction * 100, agg["mean"], yerr=agg["std"],
                    marker="o", capsize=4)
        ax.set(xlabel="Training sample size (%)", ylabel="PC skeleton edges",
               title="Sample size versus learned graph size")
        fig.tight_layout()
        fig.savefig(out / "sample_size_vs_edges.png", dpi=220)
        plt.close(fig)

    if not robustness.empty:
        agg = robustness.groupby(["model", "condition"]).AUROC.mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, group in agg.groupby("model"):
            ax.plot(group.condition, group.AUROC, marker="o", label=model)
        ax.set(ylabel="AUROC", title="Predictive robustness under test-time corruption")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "robustness_auroc.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(inference.number_of_observed_variables,
                inference.mean_inference_time_ms,
                yerr=inference.std_inference_time_ms, marker="o", capsize=3)
    ax.set(xlabel="Number of observed variables",
           ylabel="Inference time (ms)",
           title="Exact inference runtime versus evidence size")
    fig.tight_layout()
    fig.savefig(out / "inference_runtime.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CSE516 M3 experiments")
    parser.add_argument("--data", default="data/survey lung cancer.csv")
    parser.add_argument("--out", default="results_m3")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--max-cond", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.folds < 2 or args.repeats < 2:
        parser.error("--folds must be >=2 and --repeats must be >=2")
    if args.bootstrap < 1:
        parser.error("--bootstrap must be >=1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    protected = set(m2.PROTECTED_EDGES)

    print("[1/8] Loading and auditing data")
    df, data_report = m2.load_data(args.data)
    names = list(df.columns)
    card = m2.cardinalities(names)
    if df[TARGET].value_counts().min() < args.folds:
        raise ValueError("Minority class has fewer samples than the number of folds.")
    (out / "dataset_audit.json").write_text(
        json.dumps(data_report, indent=2), encoding="utf-8")

    print("[2/8] Repeated stratified predictive evaluation")
    fold_df, repeat_df, pred_df = repeated_predictive_experiments(
        df, names, card, args.folds, args.repeats, args.seed,
        args.alpha, args.max_cond, protected)
    fold_df.to_csv(out / "fold_level_metrics.csv", index=False)
    repeat_df.to_csv(out / "repeat_level_metrics.csv", index=False)
    pred_df.to_csv(out / "out_of_fold_predictions.csv", index=False)
    summary = summarize_repeats(repeat_df)
    summary.to_csv(out / "kpi_summary_mean_sd_ci95.csv", index=False)

    print("[3/8] Paired statistical comparisons")
    stats = paired_statistics(repeat_df)
    stats.to_csv(out / "paired_statistical_tests.csv", index=False)

    print("[4/8] PC alpha sensitivity and protected-edge ablation")
    structure = structure_sensitivity(
        df, [0.01, 0.05, 0.10], args.max_cond, protected, out)
    glasso_summary = graphical_lasso_comparison(df, names, out)

    print("[5/8] Bootstrap edge stability")
    bootstrap = bootstrap_edge_stability(
        df, args.alpha, args.max_cond, args.bootstrap, args.seed,
        set(), out)

    print("[6/8] Sample-size and test-time robustness")
    sample_size = sample_size_experiment(
        df, [0.20, 0.40, 0.60, 0.80, 1.00], 10,
        args.alpha, args.max_cond, args.seed, out)
    robustness = robustness_prediction_experiments(
        df, names, card, args.folds, args.seed, args.alpha,
        args.max_cond, protected, out)

    print("[7/8] Exact inference validation and runtime")
    inference, cpdag, dag, skel, ci_log = inference_experiments(
        df, names, card, args.alpha, args.max_cond, protected, out, args.seed)
    (out / "cpdag.json").write_text(json.dumps(cpdag, indent=2), encoding="utf-8")
    pd.DataFrame([{"u": u, "v": v} for u, v in skel.edges()]).to_csv(
        out / "pc_skeleton_edges.csv", index=False)
    pd.DataFrame([{"source": u, "target": v,
                   "orientation_status": (
                       "PC compelled" if [u, v] in cpdag["directed_edges"]
                       else "DAG extension only")}
                  for u, v in dag.edges()]).to_csv(
        out / "bn_dag_edges_with_orientation_status.csv", index=False)
    pd.DataFrame(ci_log).to_csv(out / "full_data_ci_test_log.csv", index=False)

    print("[8/8] Figures and reproducibility metadata")
    calibration_plot(pred_df, out)
    error_analysis(pred_df, out)
    plot_results(summary, structure, bootstrap, sample_size,
                 robustness, inference, out)
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "dataset": str(args.data), "rows_after_deduplication": len(df),
        "class_counts": {str(k): int(v) for k, v in
                         df[TARGET].value_counts().to_dict().items()},
        "seed": args.seed, "folds": args.folds, "repeats": args.repeats,
        "bootstrap_runs": args.bootstrap, "pc_alpha": args.alpha,
        "max_conditioning_set_size": args.max_cond,
        "predictive_bn_ablation": [
            "PC-learned Bayesian Network (protected edges)",
            "PC-learned Bayesian Network (plain PC)"
        ],
        "protected_edges_used_for_predictive_BN": [
            sorted(e) for e in protected],
        "bootstrap_structure_uses_protected_edges": False,
        "graphical_lasso_summary": glasso_summary,
        "statistical_test_caveat":
            "Repeated-CV comparisons are exploratory because repeats share observations.",
        "shd_note":
            "No SHD is reported on the real survey data because no ground-truth graph exists.",
        "ngm_note":
            "No NGM was implemented; literature results remain external references."
    }
    (out / "reproducibility_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nM3 experiments completed.")
    print("Output directory:", out.resolve())
    print("\nPrimary KPI summary:")
    print(summary[summary.metric.isin(["AUROC", "AUPRC", "F1_NO",
                                       "Brier_score", "Log_loss", "ECE"])]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
