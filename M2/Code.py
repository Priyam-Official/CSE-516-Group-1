import argparse, json, platform, time, warnings
import urllib.request, zipfile, io
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TARGET = "LUNG_CANCER"
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/nancyalaswad90/lung-cancer"
AGE_BINS = 3

# pairs we expect to see based on clinical intuition
SANITY_PAIRS = [("SMOKING", "YELLOW_FINGERS"), ("SMOKING", "COUGHING"),
                ("SMOKING", "CHRONIC_DISEASE"), ("COUGHING", "WHEEZING"),
                ("COUGHING", "SHORTNESS_OF_BREATH"), ("WHEEZING", "SHORTNESS_OF_BREATH"),
                ("ALCOHOL_CONSUMING", "LUNG_CANCER"), ("CHEST_PAIN", "LUNG_CANCER")]


# ---------- data ----------

def load_and_clean(path):
    if path.startswith("http://") or path.startswith("https://"):
        print("downloading dataset...")
        with urllib.request.urlopen(path) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError("no csv inside the zip")
            print("using", csvs[0])
            raw = pd.read_csv(z.open(csvs[0]))
    else:
        raw = pd.read_csv(path)

    raw.columns = [c.strip().upper().replace(" ", "_") for c in raw.columns]
    df = raw.copy()

    df["GENDER"] = df["GENDER"].astype(str).str.strip().str.upper().map({"M": 1, "F": 0})
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})

    # kaggle encodes 1=no 2=yes for the symptom columns
    for c in df.columns:
        if c not in ("GENDER", "AGE", TARGET):
            df[c] = df[c].map({1: 0, 2: 1})

    assert df.isna().sum().sum() == 0, "encoding mismatch"
    df["AGE"] = df["AGE"].astype(int)

    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    rep = {"rows_raw": int(n0), "rows_after_dedup": int(len(df)),
           "duplicates_removed": int(n0 - len(df)),
           "n_variables": int(df.shape[1]),
           "class_balance_positive": float(df[TARGET].mean()),
           "majority_class_accuracy": float(max(df[TARGET].mean(), 1 - df[TARGET].mean())),
           "age_min": int(df.AGE.min()), "age_max": int(df.AGE.max()),
           "age_mean": float(df.AGE.mean()),
           "per_variable_positive_rate": {c: float(df[c].mean()) for c in df.columns if c != "AGE"}}
    return df, rep


class Discretizer:
    # age -> tertiles, fit on train only
    def fit(self, age):
        self.cuts = np.unique(np.quantile(age, np.linspace(0, 1, AGE_BINS + 1)[1:-1]))
        return self

    def transform(self, age):
        return np.digitize(age, self.cuts, right=True)


def names_and_card(df):
    names = [c for c in df.columns if c != TARGET] + [TARGET]
    card = np.array([AGE_BINS if n == "AGE" else 2 for n in names])
    return names, card


def to_int(df, names, disc):
    X = df[names].copy()
    X["AGE"] = disc.transform(df["AGE"].values)
    return X.values.astype(np.int64)


# ---------- bayes net ----------

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


def local_bic(data, card, i, ps):
    N = data.shape[0]
    c = counts_table(data, card, i, list(ps))
    nij = c.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.where(c > 0, c * np.log(c / nij), 0.0).sum()
    return ll - 0.5 * np.log(N) * (card[i] - 1) * c.shape[0]


def _creates_cycle(parents, u, v):
    # adding u->v cycles iff v is already an ancestor of u
    stack, seen = [u], set()
    while stack:
        x = stack.pop()
        if x == v:
            return True
        for p in parents[x]:
            if p not in seen:
                seen.add(p); stack.append(p)
    return False


def hill_climb(data, card, max_indegree=3, forbidden=frozenset(), max_iter=500):
    n = data.shape[1]
    parents = [set() for _ in range(n)]
    cache = {}

    def sc(i, ps):
        key = (i, tuple(sorted(ps)))
        if key not in cache:
            cache[key] = local_bic(data, card, i, key[1])
        return cache[key]

    cur = [sc(i, set()) for i in range(n)]

    for _ in range(max_iter):
        best_d, best_move = 1e-9, None

        for u in range(n):
            for v in range(n):
                if u == v:
                    continue

                if u in parents[v]:
                    d = sc(v, parents[v] - {u}) - cur[v]
                    if d > best_d:
                        best_d, best_move = d, ("del", u, v)

                    if (v, u) not in forbidden and len(parents[u]) < max_indegree:
                        parents[v].discard(u)
                        cyc = _creates_cycle(parents, v, u)
                        parents[v].add(u)
                        if not cyc:
                            d = (sc(v, parents[v] - {u}) - cur[v]) + (sc(u, parents[u] | {v}) - cur[u])
                            if d > best_d:
                                best_d, best_move = d, ("rev", u, v)

                elif v not in parents[u]:
                    if (u, v) in forbidden or len(parents[v]) >= max_indegree:
                        continue
                    if _creates_cycle(parents, u, v):
                        continue
                    d = sc(v, parents[v] | {u}) - cur[v]
                    if d > best_d:
                        best_d, best_move = d, ("add", u, v)

        if best_move is None:
            break

        op, u, v = best_move
        if op == "add":
            parents[v].add(u)
        elif op == "del":
            parents[v].discard(u)
        else:
            parents[v].discard(u); parents[u].add(v)

        for k in (u, v):
            cur[k] = sc(k, parents[k])

    return [sorted(p) for p in parents], float(sum(cur))


def star_structure(n, target):
    # naive bayes = every feature has target as parent
    parents = [[] for _ in range(n)]
    for i in range(n):
        if i != target:
            parents[i] = [target]
    return parents


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


class BayesNet:
    def __init__(self, names, card, parents, alpha=1.0):
        self.names = names
        self.card = np.asarray(card)
        self.parents = parents
        self.alpha = alpha
        self.n = len(names)

    def fit(self, data):
        self.flat, self.factors = [], []
        for i in range(self.n):
            ps = self.parents[i]
            c = counts_table(data, self.card, i, ps) + self.alpha
            cpt = c / c.sum(1, keepdims=True)
            self.flat.append(cpt)
            shape = [self.card[p] for p in ps] + [self.card[i]]
            self.factors.append(Factor(ps + [i], cpt.reshape(shape)))
        return self

    def n_params(self):
        return int(sum((self.card[i] - 1) * np.prod([self.card[p] for p in self.parents[i]] or [1])
                       for i in range(self.n)))

    def n_edges(self):
        return int(sum(len(p) for p in self.parents))

    def bic(self, data):
        return float(self.loglik_rows(data).sum() - 0.5 * np.log(len(data)) * self.n_params())

    def loglik_rows(self, data):
        ll = np.zeros(data.shape[0])
        for i in range(self.n):
            cfg = _config_index(data, self.card, self.parents[i])
            ll += np.log(self.flat[i][cfg, data[:, i]])
        return ll

    def posterior(self, target, evidence):
        # variable elimination, greedy ordering
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
        return float(self.posterior(target, {j: int(row[j]) for j in observed})[1])

    def brute_force_posterior(self, target, evidence):
        joint = self.factors[0]
        for f in self.factors[1:]:
            joint = f_multiply(joint, f, self.card)
        t = f_reduce(joint, evidence)
        for v in [x for x in t.vars if x != target]:
            t = f_marginalize(t, v)
        p = t.t.reshape(-1)
        return p / p.sum()

    def markov_blanket(self, v):
        mb = set(self.parents[v])
        for c in range(self.n):
            if v in self.parents[c]:
                mb.add(c); mb |= set(self.parents[c])
        mb.discard(v)
        return sorted(mb)

    def edges(self):
        return [(self.names[p], self.names[i]) for i in range(self.n) for p in self.parents[i]]


def forbidden_edges(names, demo_exo=True):
    if not demo_exo:
        return frozenset()
    n = len(names)
    exo = [names.index("GENDER"), names.index("AGE")]
    return frozenset((u, e) for e in exo for u in range(n) if u != e)


def learn_bn(data, names, card, max_indegree, alpha, demo_exo=True):
    parents, _ = hill_climb(data, card, max_indegree, forbidden_edges(names, demo_exo))
    return BayesNet(names, card, parents, alpha).fit(data)


def learn_nb(data, names, card, alpha):
    return BayesNet(names, card, star_structure(len(names), len(names) - 1), alpha).fit(data)


# ---------- ci graph ----------

def learn_ci_graph(df, names, alpha=None, seed=0):
    X = StandardScaler().fit_transform(df[names].values.astype(float))
    gl = GraphicalLassoCV(cv=5, max_iter=500) if alpha is None else GraphicalLasso(alpha=alpha, max_iter=500)
    gl.fit(X)
    P = gl.precision_
    d = np.sqrt(np.diag(P))
    pcor = -P / np.outer(d, d)
    np.fill_diagonal(pcor, 0)
    pcor[np.abs(pcor) < 1e-8] = 0
    return pcor, float(getattr(gl, "alpha_", alpha))


# ---------- metrics ----------

def ece_score(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def prob_metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {"auroc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
            "aupr": average_precision_score(y, p) if len(set(y)) > 1 else np.nan,
            "accuracy": accuracy_score(y, p >= 0.5),
            "brier": brier_score_loss(y, p),
            "ece": ece_score(y, p),
            "logloss": log_loss(y, p, labels=[0, 1])}


def skeleton(edges):
    return {frozenset(e) for e in edges}


# ---------- experiments ----------

class Majority:
    def fit(self, y):
        self.p = float(np.mean(y)); return self
    def predict(self, n):
        return np.full(n, self.p)


def fit_all(tr_df, names, card, P):
    disc = Discretizer().fit(tr_df["AGE"].values)
    Xtr = to_int(tr_df, names, disc)
    bn = learn_bn(Xtr, names, card, P["max_indegree"], P["alpha"], P["demo_exo"])
    nb = learn_nb(Xtr, names, card, P["alpha"])

    feats = names[:-1]
    sc = StandardScaler().fit(tr_df[feats].values.astype(float))
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(
        sc.transform(tr_df[feats].values.astype(float)), tr_df[TARGET].values)
    maj = Majority().fit(tr_df[TARGET].values)
    return disc, bn, nb, lr, sc, maj


def cv_experiment(df, names, card, P, out):
    T = len(names) - 1
    feats = list(range(T))
    rskf = RepeatedStratifiedKFold(n_splits=P["folds"], n_repeats=P["repeats"], random_state=P["seed"])
    rows, pooled = [], {m: ([], []) for m in ("BN", "NB", "LR", "Majority")}

    for fold, (tr, te) in enumerate(rskf.split(df, df[TARGET])):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        disc, bn, nb, lr, sc, maj = fit_all(tr_df, names, card, P)
        Xte, yte = to_int(te_df, names, disc), te_df[TARGET].values

        preds, lat = {}, {}
        for tag, model in (("BN", bn), ("NB", nb)):
            t0 = time.perf_counter()
            preds[tag] = np.array([model.p_target(r, feats, T) for r in Xte])
            lat[tag] = 1000 * (time.perf_counter() - t0) / len(Xte)

        t0 = time.perf_counter()
        preds["LR"] = lr.predict_proba(sc.transform(te_df[names[:-1]].values.astype(float)))[:, 1]
        lat["LR"] = 1000 * (time.perf_counter() - t0) / len(Xte)

        preds["Majority"] = maj.predict(len(yte)); lat["Majority"] = 0.0

        jll = {"BN": bn.loglik_rows(Xte).mean(), "NB": nb.loglik_rows(Xte).mean()}

        for m, p in preds.items():
            r = {"fold": fold, "model": m, "latency_ms": lat[m],
                 "heldout_joint_ll": jll.get(m, np.nan)}
            r.update(prob_metrics(yte, p))
            rows.append(r)
            pooled[m][0].extend(yte.tolist()); pooled[m][1].extend(p.tolist())

        print(f"  fold {fold + 1}/{P['folds'] * P['repeats']}", end="\r")

    print()
    res = pd.DataFrame(rows)
    res.to_csv(out / "cv_results_per_fold.csv", index=False)
    return res, pooled


def summarize(res):
    cols = ["auroc", "aupr", "accuracy", "brier", "ece", "logloss", "heldout_joint_ll", "latency_ms"]
    out = []
    for m, g in res.groupby("model", sort=False):
        row = {"model": m}
        for c in cols:
            v = g[c].dropna().values
            if len(v) == 0:
                row[c + "_mean"] = row[c + "_sd"] = row[c + "_ci_lo"] = row[c + "_ci_hi"] = np.nan
                continue
            mu, sd = v.mean(), v.std(ddof=1)
            h = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row.update({c + "_mean": mu, c + "_sd": sd, c + "_ci_lo": mu - h, c + "_ci_hi": mu + h})
        out.append(row)
    return pd.DataFrame(out)


def stat_tests(res, n_train, n_test):
    rows = []
    for a, b in (("BN", "LR"), ("BN", "NB"), ("NB", "LR")):
        for metric in ("auroc", "brier", "logloss", "ece"):
            x = res[res.model == a].sort_values("fold")[metric].values
            y = res[res.model == b].sort_values("fold")[metric].values
            d = x - y
            try:
                w = stats.wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                w = np.nan
            J = len(d); var = d.var(ddof=1)
            tstat = d.mean() / np.sqrt((1 / J + n_test / n_train) * var) if var > 0 else np.nan
            p_nb = 2 * stats.t.sf(abs(tstat), J - 1) if np.isfinite(tstat) else np.nan
            rows.append({"A": a, "B": b, "metric": metric, "mean_diff_A_minus_B": d.mean(),
                         "wilcoxon_p": w, "corrected_t_p": p_nb})
    return pd.DataFrame(rows)


def partial_evidence_experiment(df, names, card, P, ks):
    T = len(names) - 1
    rskf = RepeatedStratifiedKFold(n_splits=P["folds"], n_repeats=P["pe_repeats"],
                                   random_state=P["seed"] + 1)
    rows = []

    for fold, (tr, te) in enumerate(rskf.split(df, df[TARGET])):
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        disc, bn, nb, lr, sc, _ = fit_all(tr_df, names, card, P)
        Xte, yte = to_int(te_df, names, disc), te_df[TARGET].values
        Zte = sc.transform(te_df[names[:-1]].values.astype(float))
        rng = np.random.default_rng(P["seed"] * 1000 + fold)

        for k in ks:
            obs = [rng.choice(T, size=k, replace=False) for _ in range(len(yte))]
            Z = Zte.copy()
            for i, o in enumerate(obs):
                m = np.ones(T, bool); m[o] = False; Z[i, m] = 0.0

            p = {"BN": np.array([bn.p_target(Xte[i], list(obs[i]), T) for i in range(len(yte))]),
                 "NB": np.array([nb.p_target(Xte[i], list(obs[i]), T) for i in range(len(yte))]),
                 "LR (mean-impute)": lr.predict_proba(Z)[:, 1]}

            for m_, pp in p.items():
                rows.append({"fold": fold, "k_observed": k, "model": m_,
                             **{kk: v for kk, v in prob_metrics(yte, pp).items()
                                if kk in ("auroc", "brier", "logloss")}})
        print(f"  pe fold {fold + 1}/{P['folds'] * P['pe_repeats']}", end="\r")

    print()
    return pd.DataFrame(rows)


def sample_efficiency_experiment(df, names, card, P, fracs):
    rskf = RepeatedStratifiedKFold(n_splits=P["folds"], n_repeats=P["se_repeats"],
                                   random_state=P["seed"] + 2)
    rows = []
    for fold, (tr, te) in enumerate(rskf.split(df, df[TARGET])):
        rng = np.random.default_rng(P["seed"] * 7 + fold)
        for fr in fracs:
            sub = rng.choice(tr, size=max(10, int(round(fr * len(tr)))), replace=False)
            tr_df, te_df = df.iloc[sub], df.iloc[te]
            if tr_df[TARGET].nunique() < 2:
                continue
            disc, bn, nb, lr, sc, _ = fit_all(tr_df, names, card, P)
            Xte, yte = to_int(te_df, names, disc), te_df[TARGET].values
            T = len(names) - 1
            p = {"BN": np.array([bn.p_target(r, list(range(T)), T) for r in Xte]),
                 "NB": np.array([nb.p_target(r, list(range(T)), T) for r in Xte]),
                 "LR": lr.predict_proba(sc.transform(te_df[names[:-1]].values.astype(float)))[:, 1]}
            for m_, pp in p.items():
                rows.append({"fold": fold, "train_fraction": fr, "model": m_, **prob_metrics(yte, pp)})
        print(f"  se fold {fold + 1}/{P['folds'] * P['se_repeats']}", end="\r")
    print()
    return pd.DataFrame(rows)


def validate_inference(df, names, card, P, n_checks=30):
    disc = Discretizer().fit(df["AGE"].values)
    X = to_int(df, names, disc)
    bn = learn_bn(X, names, card, P["max_indegree"], P["alpha"], P["demo_exo"])
    rng = np.random.default_rng(P["seed"]); T = len(names) - 1
    err = 0.0
    for _ in range(n_checks):
        k = int(rng.integers(0, T + 1)); obs = rng.choice(T, size=k, replace=False)
        row = X[rng.integers(len(X))]
        ev = {int(j): int(row[j]) for j in obs}
        err = max(err, float(np.abs(bn.posterior(T, ev) - bn.brute_force_posterior(T, ev)).max()))
    return err


def bootstrap_edges(df, names, card, P, B):
    cnt_dir, cnt_skel = {}, {}
    rng = np.random.default_rng(P["seed"])
    for b in range(B):
        bs = df.iloc[rng.integers(0, len(df), len(df))]
        disc = Discretizer().fit(bs["AGE"].values)
        bn = learn_bn(to_int(bs, names, disc), names, card, P["max_indegree"], P["alpha"], P["demo_exo"])
        for e in bn.edges():
            cnt_dir[e] = cnt_dir.get(e, 0) + 1
        for s in skeleton(bn.edges()):
            key = tuple(sorted(s)); cnt_skel[key] = cnt_skel.get(key, 0) + 1
        print(f"  bootstrap {b + 1}/{B}", end="\r")
    print()
    return ({e: c / B for e, c in cnt_dir.items()}, {e: c / B for e, c in cnt_skel.items()})


# ---------- plots ----------

def _short(n):
    return (n.replace("SHORTNESS_OF_BREATH", "SHORT.\nBREATH")
             .replace("SWALLOWING_DIFFICULTY", "SWALLOW.\nDIFF.")
             .replace("ALCOHOL_CONSUMING", "ALCOHOL")
             .replace("CHRONIC_DISEASE", "CHRONIC\nDISEASE")
             .replace("YELLOW_FINGERS", "YELLOW\nFINGERS")
             .replace("PEER_PRESSURE", "PEER\nPRESSURE")
             .replace("LUNG_CANCER", "LUNG\nCANCER")
             .replace("CHEST_PAIN", "CHEST\nPAIN"))


def draw_ego_graph(G, target, path, title, weights=None, mb=None):
    if mb is None:
        mb = set(G.predecessors(target)) | set(G.successors(target)) if G.is_directed() else set(G.neighbors(target))
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

    colors = ["#c0392b" if n == target else "#ffe066" if n in mb else "#a8d8ea" for n in G.nodes]
    edgecols = ["#8b1a10" if n == target else "#d4b106" if n in mb else "#6fa8c4" for n in G.nodes]
    widths = [0.8 + 2.2 * (weights.get(tuple(e), weights.get(tuple(reversed(e)), 0.3))) if weights else 1.2
              for e in G.edges]

    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, edge_color="#7f8c8d", alpha=0.75,
                           arrows=G.is_directed(), arrowstyle="-|>", arrowsize=14,
                           node_size=3200, connectionstyle="arc3,rad=0.06")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, edgecolors=edgecols,
                           node_size=3200, linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: _short(n) for n in G.nodes},
                            ax=ax, font_size=7.5, font_color="black")

    ax.set_title(title, fontsize=13); ax.axis("off")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=c, mec="k", ms=11, label=l) for c, l in
                       (("#c0392b", "Target"),
                        ("#ffe066", "Markov blanket / direct neighbours"),
                        ("#a8d8ea", "Other variables"))],
              loc="lower left", frameon=False)

    fig.tight_layout(); fig.savefig(path, dpi=200, facecolor=fig.get_facecolor()); plt.close(fig)


def plot_curves(df, xcol, path, xlabel, title, metric="auroc"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, g in df.groupby("model"):
        s = g.groupby(xcol)[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(s[xcol], s["mean"], yerr=s["std"], marker="o", capsize=3, label=m)
    ax.set_xlabel(xlabel); ax.set_ylabel(metric.upper()); ax.set_title(title)
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
def plot_reliability(pooled, path, bins=8):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for m in ("BN", "NB", "LR"):
        y, p = np.array(pooled[m][0]), np.array(pooled[m][1])
        idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
        xs, ys = [], []
        for b in range(bins):
            if (idx == b).sum() >= 5:
                xs.append(p[idx == b].mean()); ys.append(y[idx == b].mean())
        ax.plot(xs, ys, marker="o", label=m)
    ax.set_xlabel("Predicted P(LUNG_CANCER=1)"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (pooled CV predictions)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
def fmt(m, s, d=3):
    return "--" if not np.isfinite(m) else f"{m:.{d}f}$\\pm${s:.{d}f}"
def write_latex_tables(summ, tests, out):
    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Model & AUROC & AUPR & Acc. & Brier & ECE & Latency (ms) \\\\", "\\midrule"]
    for _, r in summ.iterrows():
        lines.append(f"{r.model} & {fmt(r.auroc_mean, r.auroc_sd)} & {fmt(r.aupr_mean, r.aupr_sd)} & "
                     f"{fmt(r.accuracy_mean, r.accuracy_sd)} & {fmt(r.brier_mean, r.brier_sd)} & "
                     f"{fmt(r.ece_mean, r.ece_sd)} & {fmt(r.latency_ms_mean, r.latency_ms_sd, 2)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_cv.tex").write_text("\n".join(lines))

    l2 = ["\\begin{tabular}{llccc}", "\\toprule",
          "A & B & Metric & Wilcoxon $p$ & Corrected-$t$ $p$ \\\\", "\\midrule"]
    for _, r in tests[tests.metric.isin(["auroc", "brier"])].iterrows():
        l2.append(f"{r.A} & {r.B} & {r.metric} & {r.wilcoxon_p:.3f} & {r.corrected_t_p:.3f} \\\\")
    l2 += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_stats.tex").write_text("\n".join(l2))
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_URL)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=100)
    ap.add_argument("--max-indegree", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--glasso-alpha", type=float, default=None)
    ap.add_argument("--no-demo-exogenous", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    P = {"seed": a.seed, "folds": a.folds,
         "repeats": 2 if a.quick else a.repeats,
         "pe_repeats": 1 if a.quick else 3,
         "se_repeats": 1 if a.quick else 3,
         "max_indegree": a.max_indegree, "alpha": a.alpha,
         "demo_exo": not a.no_demo_exogenous}
    B = 10 if a.quick else a.bootstrap
    np.random.seed(a.seed)

    print("[1/8] loading + cleaning")
    df, rep = load_and_clean(a.data)
    names, card = names_and_card(df)
    T = len(names) - 1
    json.dump(rep, open(out / "dataset_report.json", "w"), indent=2)
    print(f"   {rep['rows_raw']} -> {rep['rows_after_dedup']} rows after dedup; "
          f"P(LC=1)={rep['class_balance_positive']:.3f}; majority={rep['majority_class_accuracy']:.3f}")

    print("[2/8] full-data BN (hc + bic)")
    disc = Discretizer().fit(df["AGE"].values)
    Xall = to_int(df, names, disc)
    bn = learn_bn(Xall, names, card, P["max_indegree"], P["alpha"], P["demo_exo"])
    nvar = len(names)
    bn_info = {"n_nodes": nvar, "n_edges": bn.n_edges(),
               "density": bn.n_edges() / (nvar * (nvar - 1) / 2),
               "n_parameters": bn.n_params(), "BIC": bn.bic(Xall),
               "age_cut_points": disc.cuts.tolist(),
               "markov_blanket_of_target": [names[i] for i in bn.markov_blanket(T)]}
    pd.DataFrame(bn.edges(), columns=["parent", "child"]).to_csv(out / "bn_edges.csv", index=False)
    json.dump({names[i]: {"parents": [names[p] for p in bn.parents[i]],
                          "cpt_rows_parent_config_cols_state": np.round(bn.flat[i], 4).tolist()}
               for i in range(nvar)}, open(out / "bn_cpts.json", "w"), indent=1)
    print("   ", bn_info)

    print("[3/8] full-data CI graph (glasso)")
    pcor, lam = learn_ci_graph(df, names, a.glasso_alpha, a.seed)
    ci_edges = [(names[i], names[j], float(pcor[i, j]))
                for i in range(nvar) for j in range(i + 1, nvar) if pcor[i, j] != 0]
    pd.DataFrame(ci_edges, columns=["u", "v", "partial_corr"]).to_csv(out / "ci_edges.csv", index=False)
    ci_info = {"lambda": lam, "n_edges": len(ci_edges),
               "density": len(ci_edges) / (nvar * (nvar - 1) / 2),
               "target_neighbours": [names[j] for j in range(nvar) if pcor[T, j] != 0]}
    print("   ", ci_info)

    print("[4/8] check VE vs enumeration")
    err = validate_inference(df, names, card, P)
    print(f"   max |VE - brute force| = {err:.2e}")
    assert err < 1e-9, "VE disagrees with enumeration!"

    print("[5/8] bootstrap edge stability")
    f_dir, f_skel = bootstrap_edges(df, names, card, P, B)
    pd.DataFrame([{"u": k[0], "v": k[1], "skeleton_freq": f}
                  for k, f in sorted(f_skel.items(), key=lambda kv: -kv[1])]
                 ).to_csv(out / "bn_edge_stability.csv", index=False)

    bn_sk = skeleton(bn.edges())
    ci_sk = skeleton([(u, v) for u, v, _ in ci_edges])
    union = len(bn_sk | ci_sk)
    cmp_info = {
        "bn_edges": len(bn_sk), "ci_edges": len(ci_sk), "shared": len(bn_sk & ci_sk),
        "jaccard": len(bn_sk & ci_sk) / union if union else np.nan,
        "skeleton_SHD": len(bn_sk ^ ci_sk),
        "stable_edges_freq>=0.8": int(sum(f >= .8 for f in f_skel.values())),
        "stable_edges_freq>=0.5": int(sum(f >= .5 for f in f_skel.values())),
        "mean_freq_of_learned_edges": float(np.mean([f_skel.get(tuple(sorted(tuple(s))), 0) for s in bn_sk])) if bn_sk else np.nan,
        "sanity_pairs": {f"{u}--{v}": {"in_BN": frozenset((u, v)) in bn_sk,
                                       "in_CI": frozenset((u, v)) in ci_sk,
                                       "bootstrap_freq": f_skel.get(tuple(sorted((u, v))), 0.0)}
                         for u, v in SANITY_PAIRS if u in names and v in names}}
    json.dump({"bn": bn_info, "ci": ci_info, "comparison": cmp_info},
              open(out / "graph_summary.json", "w"), indent=2)

    print("[6/8] repeated stratified CV")
    res, pooled = cv_experiment(df, names, card, P, out)
    summ = summarize(res); summ.to_csv(out / "cv_summary.csv", index=False)
    n_te = len(df) / P["folds"]; n_tr = len(df) - n_te
    tests = stat_tests(res, n_tr, n_te); tests.to_csv(out / "stat_tests.csv", index=False)
    write_latex_tables(summ, tests, out)
    print(summ[["model", "auroc_mean", "auroc_sd", "brier_mean", "ece_mean", "latency_ms_mean"]].round(3).to_string(index=False))

    print("[7/8] partial-evidence + sample-efficiency")
    ks = [0, 3, 6, 9, 12, T]
    pe = partial_evidence_experiment(df, names, card, P, ks)
    pe.to_csv(out / "partial_evidence.csv", index=False)
    se = sample_efficiency_experiment(df, names, card, P, [0.2, 0.4, 0.6, 0.8, 1.0])
    se.to_csv(out / "sample_efficiency.csv", index=False)

    print("[8/8] figures")
    Gb = nx.DiGraph(); Gb.add_nodes_from(names); Gb.add_edges_from(bn.edges())
    draw_ego_graph(Gb, TARGET, out / "fig_bn_graph.png",
                   "Learned Bayesian Network (hill-climbing + BIC)",
                   weights={e: f_skel.get(tuple(sorted(e)), 0.3) for e in Gb.edges},
                   mb=[names[i] for i in bn.markov_blanket(T)])
    Gc = nx.Graph(); Gc.add_nodes_from(names)
    Gc.add_edges_from([(u, v) for u, v, _ in ci_edges])
    draw_ego_graph(Gc, TARGET, out / "fig_ci_graph.png",
                   "Conditional-independence graph (graphical lasso)",
                   weights={(u, v): min(1, abs(w) * 2) for u, v, w in ci_edges})
    plot_curves(pe, "k_observed", out / "fig_partial_evidence.png",
                "# observed features", "AUROC under partial evidence")
    plot_curves(se, "train_fraction", out / "fig_sample_efficiency.png",
                "fraction of training data", "Sample efficiency")
    plot_reliability(pooled, out / "fig_reliability.png")
    json.dump({"python": platform.python_version(), "numpy": np.__version__,
               "pandas": pd.__version__, "networkx": nx.__version__,
               "params": P, "bootstrap_B": B, "ve_vs_enumeration_max_abs_err": err,
               "command": vars(a)},
              open(out / "run_config.json", "w"), indent=2, default=str)
    print("done ->", out.resolve())
if __name__ == "__main__":
    main()