"""Merged pipeline: PC-learned Bayesian Network + LR / MLP / SVM baselines (from script 2)
with the graph drawing, graphical-lasso CI graph and fast variable-elimination inference
(from script 1).

Leakage-aware: age cut-points, PC structure, CPTs and all classifier preprocessing are
learned on the training fold only.

SMOKING is merged into the graph through a small set of clinically motivated, protected
edges (SMOKING-WHEEZING, SMOKING-COUGHING, SMOKING-YELLOW_FINGERS, SMOKING-CHRONIC_DISEASE).
In small samples the G-test can drop these edges because their p-values sit just above
alpha; protecting them keeps the well-established respiratory/smoking dependencies in the
learned structure while leaving every other edge to the data.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import platform
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

TARGET = "LUNG_CANCER"
SEED = 516
AGE_BINS = 3
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/nancyalaswad90/lung-cancer"
ALIASES = {"CHRONIC DISEASE": "CHRONIC_DISEASE", "ALCOHOL CONSUMING": "ALCOHOL_CONSUMING",
           "YELLOW FINGERS": "YELLOW_FINGERS", "PEER_PRESSURE": "PEER_PRESSURE",
           "SHORTNESS OF BREATH": "SHORTNESS_OF_BREATH",
           "SWALLOWING DIFFICULTY": "SWALLOWING_DIFFICULTY", "CHEST PAIN": "CHEST_PAIN"}

# ---------------------------------------------------------------------
# Clinically motivated edges that PC is not allowed to delete.
# Rationale:
#   SMOKING -> WHEEZING           (airway irritation / obstruction)
#   SMOKING -> COUGHING           (classic smoker's cough)
#   SMOKING -> YELLOW_FINGERS     (nicotine staining)
#   SMOKING -> CHRONIC_DISEASE    (smoking-related chronic conditions)
# In the Kaggle survey, G-tests on these pairs return p-values just above alpha
# (0.052-0.115 on the released file), so a plain PC run removes them. We keep them so
# the learned structure retains SMOKING's role as a driver of respiratory symptoms.
# ---------------------------------------------------------------------
PROTECTED_EDGES = {
    frozenset(("SMOKING", "WHEEZING")),
    frozenset(("SMOKING", "COUGHING")),
    frozenset(("SMOKING", "YELLOW_FINGERS")),
    frozenset(("SMOKING", "CHRONIC_DISEASE")),
}


# =====================================================================
# Data
# =====================================================================
def _read_csv(path):
    if str(path).startswith(("http://", "https://")):
        print("downloading dataset...")
        with urllib.request.urlopen(path) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError("no csv inside the zip")
            return pd.read_csv(z.open(csvs[0]))
    return pd.read_csv(path)


def load_data(path):
    raw = _read_csv(path)
    raw.columns = [ALIASES.get(c.strip().upper(), c.strip().upper().replace(" ", "_"))
                   for c in raw.columns]
    required = {"GENDER", "AGE", TARGET}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {required - set(raw.columns)}")
    df = raw.copy()
    df["GENDER"] = df.GENDER.astype(str).str.strip().str.upper().map({"M": 1, "F": 0})
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    for c in df.columns:
        if c not in ("GENDER", "AGE", TARGET):
            df[c] = pd.to_numeric(df[c], errors="coerce").map({1: 0, 2: 1})
    df["AGE"] = pd.to_numeric(df.AGE, errors="coerce")
    if df.isna().any().any():
        raise ValueError("Missing/unknown values after encoding; inspect CSV categories.")
    df["AGE"] = df.AGE.astype(int)
    # target last (the Bayesian-network code relies on this)
    df = df[[c for c in df.columns if c != TARGET] + [TARGET]]

    report = {"raw_rows": int(len(df)), "duplicate_rows": int(df.duplicated().sum()),
              "columns": list(df.columns),
              "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().items()},
              "age_min": int(df.AGE.min()), "age_max": int(df.AGE.max())}
    df = df.drop_duplicates().reset_index(drop=True)  # dedup BEFORE splitting
    report["rows_after_dedup"] = int(len(df))
    report["positive_rate"] = float(df[TARGET].mean())
    report["majority_class_accuracy"] = float(max(df[TARGET].mean(), 1 - df[TARGET].mean()))
    return df, report


def discretize_age(df, cuts=None):
    """Tertiles of AGE. Pass `cuts` (learned on train) to transform test data."""
    out = df.copy()
    if cuts is None:
        cuts = np.unique(np.quantile(out["AGE"], np.linspace(0, 1, AGE_BINS + 1)[1:-1]))
    out["AGE"] = np.digitize(out["AGE"].to_numpy(), cuts, right=True).astype(int)
    return out, [float(x) for x in cuts]


def to_int(df_disc, names):
    return df_disc[names].values.astype(np.int64)


def cardinalities(names):
    return np.array([AGE_BINS if n == "AGE" else 2 for n in names])


# =====================================================================
# PC algorithm (script 2)
# =====================================================================
def ci_test(df, x, y, conditioning):
    """Stratified G-test CI p-value (statistics and dof summed over strata)."""
    statistic, dof_total = 0.0, 0
    groups = [(None, df)] if not conditioning else df.groupby(
        list(conditioning), dropna=False, observed=True)
    for _, group in groups:
        table = pd.crosstab(group[x], group[y], dropna=False)
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        try:
            g, _, dof, _ = chi2_contingency(table, correction=False, lambda_="log-likelihood")
        except ValueError:
            continue
        statistic += float(g)
        dof_total += int(dof)
    if dof_total == 0:
        return 1.0
    return float(chi2.sf(statistic, dof_total))


def pc_skeleton(df, alpha=0.05, max_cond=3, protected=None):
    """PC skeleton with an optional protected-edge set.

    `protected` is an iterable of frozenset({u, v}). Any pair in that set is never
    removed during skeleton discovery, regardless of the G-test p-value. This lets us
    retain clinically motivated dependencies (e.g. SMOKING-WHEEZING) that sit near the
    alpha threshold on a small sample. The decision is logged with `protected=True`
    so the audit trail remains complete.
    """
    protected = set(protected or [])
    nodes = list(df.columns)
    adj = {n: set(nodes) - {n} for n in nodes}
    seps, log = {}, []
    for level in range(max_cond + 1):
        adj_level = {n: set(v) for n, v in adj.items()}
        pairs = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:] if b in adj_level[a]]
        for x, y in pairs:
            if y not in adj[x]:
                continue
            is_protected = frozenset((x, y)) in protected
            removed, seen = False, set()
            for pool in (sorted(adj_level[x] - {y}), sorted(adj_level[y] - {x})):
                if len(pool) < level:
                    continue
                for cond in itertools.combinations(pool, level):
                    key = frozenset(cond)
                    if key in seen:
                        continue
                    seen.add(key)
                    p = ci_test(df, x, y, cond)
                    log.append({"x": x, "y": y, "conditioning": list(cond),
                                "p_value": p, "alpha": alpha, "level": level,
                                "protected": is_protected})
                    if p > alpha:
                        if is_protected:
                            # Record the near-independent test but keep the edge.
                            continue
                        adj[x].discard(y); adj[y].discard(x)
                        seps[tuple(sorted((x, y)))] = list(cond)
                        removed = True
                        break
                if removed:
                    break
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for i, x in enumerate(nodes):
        for y in nodes[i + 1:]:
            if y in adj[x]:
                g.add_edge(x, y)
    return g, seps, log


def orient_cpdag(skel, seps):
    """Collider orientation + Meek R1/R2/R3, then a deterministic DAG extension."""
    und = {frozenset(e) for e in skel.edges()}
    directed, nodes, conflicts = set(), list(skel.nodes()), []
    order = {n: i for i, n in enumerate(nodes)}

    def is_und(a, b):
        return frozenset((a, b)) in und

    def orient(a, b):
        e = frozenset((a, b))
        if e not in und:
            return False
        dg = nx.DiGraph(); dg.add_nodes_from(nodes); dg.add_edges_from(directed)
        if nx.has_path(dg, b, a):
            conflicts.append({"attempted": [a, b], "reason": "would_create_directed_cycle"})
            return False
        und.remove(e); directed.add((a, b))
        return True

    def dpath(a, b):
        dg = nx.DiGraph(); dg.add_nodes_from(nodes); dg.add_edges_from(directed)
        return nx.has_path(dg, a, b)

    def sorted_und():
        pairs = []
        for e in und:
            a, b = sorted(e, key=order.get)
            pairs.append((a, b))
        return sorted(pairs, key=lambda p: (order[p[0]], order[p[1]]))

    for z in nodes:  # v-structures
        nbr = list(skel.neighbors(z))
        for i, x in enumerate(nbr):
            for y in nbr[i + 1:]:
                if skel.has_edge(x, y):
                    continue
                if z not in seps.get(tuple(sorted((x, y))), []):
                    orient(x, z); orient(y, z)

    changed, passes = True, 0
    while changed and passes < 100:
        changed, passes = False, passes + 1
        for a, b in list(directed):                     # R1
            for c in nodes:
                if c != a and is_und(b, c) and not skel.has_edge(a, c):
                    changed |= orient(b, c)
        for a, c in sorted_und():                       # R2
            if not is_und(a, c):
                continue
            if dpath(a, c):
                changed |= orient(a, c)
            elif dpath(c, a):
                changed |= orient(c, a)
        for a, d in sorted_und():                       # R3 (both directions of edge)
            for aa, dd in ((a, d), (d, a)):
                if not is_und(aa, dd):
                    continue
                cands = [b for b in nodes if is_und(aa, b) and (b, dd) in directed]
                for i, b in enumerate(cands):
                    for c in cands[i + 1:]:
                        if not skel.has_edge(b, c):
                            changed |= orient(aa, dd)

    cpdag = {"directed_edges": sorted([list(e) for e in directed]),
             "undirected_edges": sorted([sorted(e) for e in und]),
             "orientation_conflicts": conflicts}

    dag = nx.DiGraph(); dag.add_nodes_from(nodes); dag.add_edges_from(directed)
    if not nx.is_directed_acyclic_graph(dag):
        raise RuntimeError("PC orientation produced a directed cycle.")
    for a, b in sorted_und():
        added = False
        for u, v in ((a, b), (b, a)):
            dag.add_edge(u, v)
            if nx.is_directed_acyclic_graph(dag):
                added = True
                break
            dag.remove_edge(u, v)
        if not added:
            raise RuntimeError(f"Could not extend CPDAG edge {a}-{b} to a DAG")
    return cpdag, dag


# =====================================================================
# Discrete Bayesian network with variable elimination (script 1 machinery)
# =====================================================================
class Factor:
    def __init__(self, vars_, table):
        self.vars, self.t = list(vars_), np.asarray(table, dtype=float)


def f_multiply(a, b, card):
    vs = a.vars + [v for v in b.vars if v not in a.vars]

    def expand(f):
        order = [v for v in vs if v in f.vars]
        t = np.transpose(f.t, [f.vars.index(v) for v in order])
        return t.reshape([card[v] if v in f.vars else 1 for v in vs])

    return Factor(vs, expand(a) * expand(b))


def f_marginalize(f, v):
    return Factor([x for x in f.vars if x != v], f.t.sum(axis=f.vars.index(v)))


def f_reduce(f, evidence):
    idx = tuple(evidence[v] if v in evidence else slice(None) for v in f.vars)
    return Factor([v for v in f.vars if v not in evidence], f.t[idx])


def _config_index(data, card, ps):
    cfg = np.zeros(data.shape[0], dtype=np.int64)
    for p in ps:
        cfg = cfg * card[p] + data[:, p]
    return cfg


def counts_table(data, card, i, ps):
    q = int(np.prod(card[list(ps)])) if ps else 1
    r = card[i]
    cfg = _config_index(data, card, ps)
    return np.bincount(cfg * r + data[:, i], minlength=q * r).reshape(q, r).astype(float)


class DiscreteBN:
    def __init__(self, names, card, dag, alpha=1.0):
        self.names, self.card, self.alpha = names, np.asarray(card), alpha
        self.n = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        self.parents = [sorted(idx[p] for p in dag.predecessors(nm)) for nm in names]

    def fit(self, data):
        self.flat, self.factors = [], []
        for i in range(self.n):
            ps = self.parents[i]
            c = counts_table(data, self.card, i, ps) + self.alpha
            cpt = c / c.sum(1, keepdims=True)
            self.flat.append(cpt)
            shape = [int(self.card[p]) for p in ps] + [int(self.card[i])]
            self.factors.append(Factor(ps + [i], cpt.reshape(shape)))
        return self

    def posterior(self, target, evidence):
        """P(target | evidence) by variable elimination. evidence: {var_index: state}."""
        factors = [f_reduce(F, evidence) for F in self.factors]
        hidden = [v for v in range(self.n) if v != target and v not in evidence]
        while hidden:
            def scope_size(v):
                s = set()
                for f in factors:
                    if v in f.vars:
                        s |= set(f.vars)
                return len(s)

            v = min(hidden, key=scope_size)
            involved = [f for f in factors if v in f.vars]
            rest = [f for f in factors if v not in f.vars]
            if involved:
                prod = involved[0]
                for f in involved[1:]:
                    prod = f_multiply(prod, f, self.card)
                rest.append(f_marginalize(prod, v))
            factors = rest
            hidden.remove(v)
        prod = factors[0]
        for f in factors[1:]:
            prod = f_multiply(prod, f, self.card)
        p = prod.t.reshape(-1)
        return p / p.sum()

    def p_target(self, row, observed, target):
        return float(self.posterior(target, {int(j): int(row[j]) for j in observed})[1])

    def brute_force_posterior(self, target, evidence):
        joint = self.factors[0]
        for f in self.factors[1:]:
            joint = f_multiply(joint, f, self.card)
        t = f_reduce(joint, evidence)
        for v in [x for x in t.vars if x != target]:
            t = f_marginalize(t, v)
        p = t.t.reshape(-1)
        return p / p.sum()


def markov_blanket(dag, node):
    mb = set(dag.predecessors(node)) | set(dag.successors(node))
    for child in dag.successors(node):
        mb |= set(dag.predecessors(child))
    mb.discard(node)
    return sorted(mb)


def learn_bn(df, names, card, alpha_pc, max_cond, alpha_cpt=1.0, protected=None):
    """Full PC -> CPDAG -> DAG -> CPT pipeline on a (training) dataframe."""
    disc, cuts = discretize_age(df)
    skel, seps, log = pc_skeleton(disc, alpha_pc, max_cond, protected=protected)
    cpdag, dag = orient_cpdag(skel, seps)
    bn = DiscreteBN(names, card, dag, alpha_cpt).fit(to_int(disc, names))
    return bn, disc, cuts, skel, seps, log, cpdag, dag


def validate_inference(bn, X, target, n_checks=20, seed=SEED):
    rng = np.random.default_rng(seed)
    err = 0.0
    for _ in range(n_checks):
        k = int(rng.integers(0, target + 1))
        obs = rng.choice(target, size=k, replace=False)
        row = X[rng.integers(len(X))]
        ev = {int(j): int(row[j]) for j in obs}
        err = max(err, float(np.abs(bn.posterior(target, ev) - bn.brute_force_posterior(target, ev)).max()))
    return err


# =====================================================================
# Conditional-independence graph (graphical lasso, script 1)
# =====================================================================
def learn_ci_graph(df, names, alpha=None):
    X = StandardScaler().fit_transform(df[names].values.astype(float))
    gl = GraphicalLassoCV(cv=5, max_iter=500) if alpha is None else GraphicalLasso(alpha=alpha, max_iter=500)
    gl.fit(X)
    P = gl.precision_
    d = np.sqrt(np.diag(P))
    pcor = -P / np.outer(d, d)
    np.fill_diagonal(pcor, 0)
    pcor[np.abs(pcor) < 1e-8] = 0
    return pcor, float(getattr(gl, "alpha_", alpha))


# =====================================================================
# Classifiers (script 2) + BN evaluated on identical folds
# =====================================================================
def evaluate_classifiers(df, names, card, folds=5, seed=SEED, alpha=0.05, max_cond=3,
                         protected=None):
    X = df.drop(columns=[TARGET]).reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    binary = [c for c in X.columns if c != "AGE"]
    preprocessor = ColumnTransformer([
        ("age", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), ["AGE"]),
        ("binary", Pipeline([("impute", SimpleImputer(strategy="most_frequent"))]), binary)])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced",
                                                  random_state=seed),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01,
                                              early_stopping=True, max_iter=1200,
                                              random_state=seed),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True,
                         random_state=seed)}
    T = len(names) - 1
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []

    def record(name, fold, y_true, prob):
        pred = (prob >= 0.5).astype(int)
        rows.append({"model": name, "fold": fold,
                     "AUROC": roc_auc_score(y_true, prob),
                     "AUPRC": average_precision_score(y_true, prob),
                     "F1_NO": f1_score(y_true, pred, pos_label=0, zero_division=0),
                     "Brier_score": brier_score_loss(y_true, prob)})

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr].copy(), X.iloc[te].copy()
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        for name, model in models.items():
            pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
            pipe.fit(X_tr, y_tr)
            record(name, fold, y_te, pipe.predict_proba(X_te)[:, 1])

        train_data = pd.concat([X_tr, y_tr.rename(TARGET)], axis=1)[names]
        bn, _, cuts, *_ = learn_bn(train_data, names, card, alpha, max_cond,
                                   protected=protected)
        test_disc, _ = discretize_age(X_te, cuts=cuts)
        Xte_int = test_disc[names[:-1]].values.astype(np.int64)
        full = np.column_stack([Xte_int, np.zeros(len(Xte_int), dtype=np.int64)])
        prob = np.array([bn.p_target(r, range(T), T) for r in full])
        record("PC-learned Bayesian Network", fold, y_te, prob)
        print(f"  fold {fold}/{folds}", end="\r")
    print()

    fold_results = pd.DataFrame(rows)
    summ = []
    label = {"F1_NO": "F1 (NO class)", "Brier_score": "Brier score"}
    for model_name, g in fold_results.groupby("model", sort=False):
        for metric in ["AUROC", "AUPRC", "F1_NO", "Brier_score"]:
            v = g[metric].to_numpy(dtype=float)
            summ.append({"model": model_name, "metric": label.get(metric, metric),
                         "mean": float(v.mean()),
                         "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                         "folds": int(len(v))})
    return pd.DataFrame(summ), fold_results


# =====================================================================
# Bootstrap edge stability (optional)
# =====================================================================
def bootstrap_skeleton(df, alpha, max_cond, B, seed=SEED, protected=None):
    rng = np.random.default_rng(seed)
    cnt = {}
    for b in range(B):
        bs = df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True)
        disc, _ = discretize_age(bs)
        skel, _, _ = pc_skeleton(disc, alpha, max_cond, protected=protected)
        for e in skel.edges():
            k = tuple(sorted(e))
            cnt[k] = cnt.get(k, 0) + 1
        print(f"  bootstrap {b + 1}/{B}", end="\r")
    print()
    return {k: v / B for k, v in cnt.items()}


# =====================================================================
# Graph drawing (script 1)
# =====================================================================
def _short(n):
    return (n.replace("SHORTNESS_OF_BREATH", "SHORT.\nBREATH")
             .replace("SWALLOWING_DIFFICULTY", "SWALLOW.\nDIFF.")
             .replace("ALCOHOL_CONSUMING", "ALCOHOL")
             .replace("CHRONIC_DISEASE", "CHRONIC\nDISEASE")
             .replace("YELLOW_FINGERS", "YELLOW\nFINGERS")
             .replace("PEER_PRESSURE", "PEER\nPRESSURE")
             .replace("LUNG_CANCER", "LUNG\nCANCER")
             .replace("CHEST_PAIN", "CHEST\nPAIN"))


def draw_ego_graph(G, target, path, title, weights=None, mb=None, protected=None):
    if mb is None:
        mb = (set(G.predecessors(target)) | set(G.successors(target))) if G.is_directed() \
            else set(G.neighbors(target))
    mb = set(mb) - {target}

    U = G.to_undirected()
    dist = nx.single_source_shortest_path_length(U, target)
    maxd = max(dist.values())
    shells = [[target]] + [sorted(n for n, d in dist.items() if d == h) for h in range(1, maxd + 1)]
    iso = sorted(n for n in G.nodes if n not in dist)
    if iso:
        shells.append(iso)
    shells = [s for s in shells if s]

    pos = nx.shell_layout(G, nlist=shells, scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#ecf0f5")
    ax.set_facecolor("#ecf0f5")

    protected = set(protected or [])
    colors = ["#c0392b" if n == target else "#ffe066" if n in mb else "#a8d8ea" for n in G.nodes]
    edgecols = ["#8b1a10" if n == target else "#d4b106" if n in mb else "#6fa8c4" for n in G.nodes]
    # Protected (SMOKING-merged) edges drawn thicker and in a distinct colour.
    def is_protected(e):
        return frozenset((e[0], e[1])) in protected
    if weights:
        widths = [0.8 + 2.2 * weights.get(tuple(e), weights.get(tuple(reversed(e)), 0.3))
                  for e in G.edges]
    else:
        widths = [3.0 if is_protected(e) else 1.2 for e in G.edges]
    ecol = ["#c0392b" if is_protected(e) else "#7f8c8d" for e in G.edges]

    if G.number_of_edges():
        nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color=ecol, alpha=0.85,
                               arrows=G.is_directed(), arrowstyle="-|>", arrowsize=14,
                               node_size=3200, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, edgecolors=edgecols,
                           node_size=3200, linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: _short(n) for n in G.nodes}, ax=ax,
                            font_size=7.5, font_color="black")
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=11, label=l)
               for c, l in (("#c0392b", "Target"),
                            ("#ffe066", "Markov blanket / direct neighbours"),
                            ("#a8d8ea", "Other variables"))]
    if protected:
        handles.append(Line2D([], [], color="#c0392b", lw=2.5,
                              label="SMOKING protected (clinically merged)"))
    ax.legend(handles=handles, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/survey lung cancer.csv",
                    help="CSV path, or the Kaggle zip URL (pass --data URL)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.05, help="PC significance level")
    ap.add_argument("--max-cond", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="# bootstrap PC runs for edge-stability line widths (0 = off)")
    ap.add_argument("--glasso-alpha", type=float, default=None)
    ap.add_argument("--skip-classifiers", action="store_true")
    ap.add_argument("--no-protect-smoking", action="store_true",
                    help="Disable SMOKING clinical-edge protection (plain PC run)")
    a = ap.parse_args()
    if not 0 < a.alpha < 1:
        ap.error("--alpha must be between 0 and 1")
    if a.max_cond < 0:
        ap.error("--max-cond must be non-negative")
    if a.folds < 2:
        ap.error("--folds must be at least 2")

    protected = set() if a.no_protect_smoking else set(PROTECTED_EDGES)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/7] loading + cleaning")
    df, report = load_data(a.data)
    names = list(df.columns)
    card = cardinalities(names)
    T = len(names) - 1
    nvar = len(names)
    print(f"   {report['raw_rows']} -> {report['rows_after_dedup']} rows; "
          f"P(LC=1)={report['positive_rate']:.3f}")

    print("[2/7] full-data PC Bayesian network (descriptive)")
    bn, disc, cuts, skel, seps, ci_log, cpdag, dag = learn_bn(
        df, names, card, a.alpha, a.max_cond, protected=protected)
    Xall = to_int(disc, names)
    mb = markov_blanket(dag, TARGET)
    compelled = {tuple(e) for e in cpdag["directed_edges"]}
    retained = sorted([sorted(e) for e in PROTECTED_EDGES
                       if e.issubset({tuple(sorted(ed)) for ed in skel.edges()})
                       or any(set(e) == set(ed) for ed in skel.edges())])
    report.update({
        "age_cut_points_full_data_descriptive_only": cuts, "pc_alpha": a.alpha,
        "pc_max_conditioning_set": a.max_cond, "ci_tests": len(ci_log),
        "skeleton_edges": skel.number_of_edges(), "skeleton_density": nx.density(skel),
        "cpdag_directed_edges": len(cpdag["directed_edges"]),
        "cpdag_undirected_edges": len(cpdag["undirected_edges"]),
        "dag_edges": dag.number_of_edges(), "dag_acyclic": nx.is_directed_acyclic_graph(dag),
        "markov_blanket_of_target": mb,
        "smoking_edge_protection_enabled": bool(protected),
        "smoking_protected_edges": [sorted(e) for e in PROTECTED_EDGES] if protected else [],
        "smoking_edges_retained": retained,
        "dag_extension_policy": "PC-compelled orientations kept; remaining CPDAG edges "
                                "oriented deterministically. Extension-only directions are "
                                "not identified by PC and must not be read causally."})
    (out / "dataset_and_structure_summary.json").write_text(json.dumps(report, indent=2))
    (out / "pc_cpdag.json").write_text(json.dumps(cpdag, indent=2))
    pd.DataFrame(ci_log).to_csv(out / "ci_test_log.csv", index=False)
    pd.DataFrame([{"source": u, "target": v} for u, v in skel.edges()]).to_csv(
        out / "pc_skeleton_edges.csv", index=False)
    pd.DataFrame([{"source": u, "target": v,
                   "orientation_status": "PC compelled" if (u, v) in compelled else "DAG extension only"}
                  for u, v in dag.edges()]).to_csv(out / "bn_dag_edges.csv", index=False)
    (out / "bn_cpts.json").write_text(json.dumps(
        {names[i]: {"parents": [names[p] for p in bn.parents[i]],
                    "cpt_rows_parent_config_cols_state": np.round(bn.flat[i], 4).tolist()}
         for i in range(nvar)}, indent=1))
    print("   edges:", dag.number_of_edges(), "| Markov blanket:", mb)
    print("   smoking protected:", bool(protected),
          "| retained:", [tuple(sorted(e)) for e in protected
                          if any(set(e) == set(ed) for ed in skel.edges())])

    print("[3/7] graphical-lasso CI graph")
    pcor, lam = learn_ci_graph(df, names, a.glasso_alpha)
    ci_edges = [(names[i], names[j], float(pcor[i, j]))
                for i in range(nvar) for j in range(i + 1, nvar) if pcor[i, j] != 0]
    pd.DataFrame(ci_edges, columns=["u", "v", "partial_corr"]).to_csv(out / "ci_edges.csv", index=False)
    print(f"   lambda={lam:.4f}, edges={len(ci_edges)}")

    print("[4/7] checking variable elimination vs brute-force enumeration")
    err = validate_inference(bn, Xall, T)
    print(f"   max |VE - brute force| = {err:.2e}")
    assert err < 1e-9, "VE disagrees with enumeration!"

    examples = [{"ALLERGY": 1}, {"ALLERGY": 1, "ALCOHOL_CONSUMING": 1},
                {"SMOKING": 1}, {"SMOKING": 1, "WHEEZING": 1, "COUGHING": 1},
                {"ALLERGY": 0, "ALCOHOL_CONSUMING": 0, "SWALLOWING_DIFFICULTY": 0}]
    inf = []
    for ev in examples:
        try:
            p = bn.posterior(T, {names.index(k): v for k, v in ev.items()})
            inf.append({"evidence": ev, "posterior": {"0": float(p[0]), "1": float(p[1])},
                        "warning": "Descriptive full-data fit; not a held-out estimate."})
        except (KeyError, ValueError) as e:
            inf.append({"evidence": ev, "error": str(e)})
    (out / "inference_examples.json").write_text(json.dumps(inf, indent=2))

    freq = {}
    if a.bootstrap > 0:
        print("[5/7] bootstrap PC edge stability")
        freq = bootstrap_skeleton(df, a.alpha, a.max_cond, a.bootstrap, protected=protected)
        pd.DataFrame([{"u": k[0], "v": k[1], "skeleton_freq": f}
                      for k, f in sorted(freq.items(), key=lambda kv: -kv[1])]
                     ).to_csv(out / "pc_edge_stability.csv", index=False)
    else:
        print("[5/7] bootstrap skipped (--bootstrap 0)")

    if not a.skip_classifiers:
        print("[6/7] cross-validated classifiers (LR / MLP / SVM / BN)")
        summary, fold_metrics = evaluate_classifiers(df, names, card, folds=a.folds,
                                                     alpha=a.alpha, max_cond=a.max_cond,
                                                     protected=protected)
        summary.to_csv(out / "classifier_metrics.csv", index=False)
        fold_metrics.to_csv(out / "classifier_fold_metrics.csv", index=False)
        pivot = summary.pivot(index="model", columns="metric", values="mean")
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel("Mean cross-validation score")
        ax.set_title("Classifier KPIs: mean across identical stratified folds")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(out / "classifier_metrics.png", dpi=220)
        plt.close()
        print(summary.pivot(index="model", columns="metric", values="mean").round(3).to_string())
    else:
        print("[6/7] classifiers skipped")

    print("[7/7] figures")
    wdir = {e: freq.get(tuple(sorted(e)), 0.3) for e in dag.edges} if freq else None
    draw_ego_graph(dag, TARGET, out / "fig_bn_graph.png",
                   "PC-learned Bayesian Network (DAG extension, SMOKING merged)",
                   weights=wdir, mb=mb, protected=protected)
    Gs = nx.Graph(); Gs.add_nodes_from(names); Gs.add_edges_from(skel.edges())
    draw_ego_graph(Gs, TARGET, out / "fig_pc_skeleton.png",
                   "PC conditional-independence skeleton (undirected, SMOKING merged)",
                   weights={tuple(sorted(e)): freq.get(tuple(sorted(e)), 0.3) for e in skel.edges()}
                   if freq else None,
                   protected=protected)
    Gc = nx.Graph(); Gc.add_nodes_from(names)
    Gc.add_edges_from([(u, v) for u, v, _ in ci_edges])
    draw_ego_graph(Gc, TARGET, out / "fig_ci_graph.png",
                   "Conditional-independence graph (graphical lasso)",
                   weights={(u, v): min(1, abs(w) * 2) for u, v, w in ci_edges},
                   protected=protected)

    (out / "run_config.json").write_text(json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "networkx": nx.__version__, "ve_vs_enumeration_max_abs_err": err,
         "protected_smoking_edges": [sorted(e) for e in protected],
         "command": vars(a)}, indent=2, default=str))
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()