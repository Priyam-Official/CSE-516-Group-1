"""Targeted M4 extensions for the CSE516 lung-cancer PGM project.

This module is invoked by main.py as the M4 stage. It does not overwrite M3
outputs. It regenerates paired tests from repeat-level metrics, validates exact
inference for evidence sizes 0-15, computes held-out joint BN NLL, and compares
BN marginalization with imputation under test-time MCAR missingness.
"""
from __future__ import annotations
import argparse, json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import Code as m2

TARGET = m2.TARGET
SEED = 516


def paired_tests(repeat_csv: Path, out: Path):
    df = pd.read_csv(repeat_csv)
    # Compare every model with each BN variant; exact model names are taken from
    # the actual M3 output, avoiding the old unmatched default reference string.
    refs = [x for x in df.model.unique() if x in {
        'PC-learned Bayesian Network (plain PC)',
        'PC-learned Bayesian Network (protected edges)'}]
    rows=[]
    for ref in refs:
        for model in sorted(set(df.model)-{ref}):
            for metric in ['AUROC','AUPRC','F1_NO','Brier_score','Log_loss','ECE','Accuracy']:
                w=df.pivot(index='repeat',columns='model',values=metric)[[model,ref]].dropna()
                a,b=w[model].to_numpy(float),w[ref].to_numpy(float)
                if len(a)<2: continue
                try: tp=float(ttest_rel(a,b).pvalue)
                except Exception: tp=float('nan')
                try: wp=float(wilcoxon(a,b,zero_method='wilcox',method='auto').pvalue)
                except ValueError: wp=1.0
                rows.append({'reference_model':ref,'comparison_model':model,'metric':metric,
                    'n_paired_repeats':len(a),'mean_difference_model_minus_reference':float(np.mean(a-b)),
                    'paired_t_test_p':tp,'wilcoxon_p':wp,
                    'interpretation_note':'Exploratory only: repeated CV runs reuse the same dataset; five repeats do not provide independent external samples.'})
    result=pd.DataFrame(rows)
    # Holm adjustment separately within each reference/metric family.
    result['wilcoxon_holm_p']=np.nan
    for _, idx in result.groupby(['reference_model','metric']).groups.items():
        vals=result.loc[idx,'wilcoxon_p'].to_numpy(float)
        order=np.argsort(vals); m=len(vals); adj=np.empty(m); running=0.0
        for rank,pos in enumerate(order):
            running=max(running,(m-rank)*vals[pos]); adj[pos]=min(1.0,running)
        result.loc[idx,'wilcoxon_holm_p']=adj
    result.to_csv(out/'paired_statistical_tests_recomputed.csv',index=False)
    return result


def brute_force_all(bn, target, evidence):
    """Use BN's explicit full-joint enumeration as independent check of VE."""
    return bn.brute_force_posterior(target,evidence)


def inference_validation(df,names,card,alpha,max_cond,out,seed,queries_per_size=3):
    disc,_=m2.discretize_age(df)
    skel,seps,_=m2.pc_skeleton(disc,alpha=alpha,max_cond=max_cond,protected=set())
    cpdag,dag=m2.orient_cpdag(skel,seps)
    X=m2.to_int(disc,names); bn=m2.DiscreteBN(names,card,dag,alpha=1.0).fit(X)
    target=names.index(TARGET); rng=np.random.default_rng(seed)
    rows=[]
    for k in range(target+1):
        errs=[]; ve_ms=[]; brute_ms=[]
        for q in range(queries_per_size):
            observed=rng.choice(target,size=k,replace=False)
            row=X[int(rng.integers(len(X)))]
            evidence={int(j):int(row[j]) for j in observed}
            t0=time.perf_counter(); pve=bn.posterior(target,evidence); ve_ms.append((time.perf_counter()-t0)*1000)
            t0=time.perf_counter(); pbf=brute_force_all(bn,target,evidence); brute_ms.append((time.perf_counter()-t0)*1000)
            errs.append(float(np.max(np.abs(pve-pbf))))
        rows.append({'evidence_variables':k,'queries':queries_per_size,
            'max_absolute_error':max(errs),'mean_absolute_error':float(np.mean(errs)),
            'mean_variable_elimination_ms':float(np.mean(ve_ms)),
            'mean_brute_force_ms':float(np.mean(brute_ms))})
        print(f'Inference validation: {k}/{target} evidence variables',flush=True)
    result=pd.DataFrame(rows); result.to_csv(out/'inference_validation_0_15.csv',index=False)
    (out/'inference_validation_graph.json').write_text(json.dumps({'nodes':names,'skeleton_edges':[list(e) for e in skel.edges()], 'cpdag':cpdag,'dag_edges':[list(e) for e in dag.edges()]},indent=2),encoding='utf-8')
    return result


def joint_log_probability(bn, data):
    """Log P(x) from learned BN CPTs, one complete categorical row at a time."""
    data=np.asarray(data,dtype=np.int64); total=np.zeros(len(data),dtype=float)
    for i, parents in enumerate(bn.parents):
        if parents:
            cfg=np.zeros(len(data),dtype=np.int64)
            for p in parents: cfg=cfg*int(bn.card[p])+data[:,p]
        else: cfg=np.zeros(len(data),dtype=np.int64)
        probs=bn.flat[i][cfg,data[:,i]]
        total += np.log(np.clip(probs,1e-15,1.0))
    return total


def single_holdout_bn_extensions(df,names,card,alpha,max_cond,seed,out):
    """One explicitly labelled 80/20 holdout: joint NLL and missing-evidence tests.

    This is supplementary evidence, not a replacement for the repeated-CV
    predictive results already reported in M3.
    """
    from sklearn.model_selection import train_test_split
    X=df.drop(columns=[TARGET]).reset_index(drop=True); y=df[TARGET].astype(int).reset_index(drop=True)
    tr,te=train_test_split(np.arange(len(df)),test_size=0.20,random_state=seed,stratify=y)
    Xtr,Xte=X.iloc[tr].copy(),X.iloc[te].copy(); ytr,yte=y.iloc[tr],y.iloc[te]
    train=pd.concat([Xtr.reset_index(drop=True),ytr.reset_index(drop=True).rename(TARGET)],axis=1)[names]
    print('Fitting one training-only PC-BN for supplementary 80/20 holdout...',flush=True)
    bn,_,cuts,*_=m2.learn_bn(train,names,card,alpha,max_cond,protected=set())
    test_disc,_=m2.discretize_age(Xte,cuts=cuts)
    test_full=np.column_stack([test_disc[names[:-1]].to_numpy(dtype=np.int64),yte.to_numpy(dtype=np.int64)])
    ll=joint_log_probability(bn,test_full)
    nll=pd.DataFrame([{'split':'stratified 80/20 holdout','train_n':len(tr),'test_n':len(te),
        'joint_nll_per_sample':float(-ll.mean()),'joint_log_likelihood_per_sample':float(ll.mean()),
        'total_joint_nll':float(-ll.sum())}])
    nll.to_csv(out/'heldout_joint_bn_nll.csv',index=False)

    base=test_disc[names[:-1]].to_numpy(dtype=np.int64); target=names.index(TARGET)
    rows=[]; fills={c:(float(Xtr[c].median()) if c=='AGE' else float(Xtr[c].mode().iloc[0])) for c in Xtr.columns}
    for rate in [0.10,0.20]:
        rng=np.random.default_rng(seed+int(rate*100))
        mask=rng.random(base.shape)<rate
        Ximp=Xte.copy()
        for j,c in enumerate(names[:-1]): Ximp.loc[mask[:,j],c]=fills[c]
        imp_disc,_=m2.discretize_age(Ximp,cuts=cuts); imp_arr=imp_disc[names[:-1]].to_numpy(dtype=np.int64)
        p_marg=[]; p_imp=[]
        for i in range(len(te)):
            evidence={j:int(base[i,j]) for j in range(base.shape[1]) if not mask[i,j]}
            p_marg.append(float(bn.posterior(target,evidence)[1]))
            evidence_imp={j:int(imp_arr[i,j]) for j in range(base.shape[1])}
            p_imp.append(float(bn.posterior(target,evidence_imp)[1]))
        for method,prob in [('BN marginalization (observed evidence only)',p_marg),('BN after train-statistic imputation',p_imp)]:
            prob=np.clip(np.asarray(prob),1e-7,1-1e-7)
            rows.append({'split':'stratified 80/20 holdout','missing_rate':rate,'method':method,
                'AUROC':float(roc_auc_score(yte,prob)),'AUPRC':float(average_precision_score(yte,prob)),
                'Brier_score':float(brier_score_loss(yte,prob)),
                'Log_loss':float(log_loss(yte,prob,labels=[0,1])),'n_test':len(te)})
    miss=pd.DataFrame(rows); miss.to_csv(out/'missing_evidence_marginalization_vs_imputation.csv',index=False)
    return nll,miss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/survey lung cancer.csv')
    ap.add_argument('--m3-results',default='results_m3')
    ap.add_argument('--out',default='results_m4')
    ap.add_argument('--seed',type=int,default=SEED)
    ap.add_argument('--alpha',type=float,default=.05)
    ap.add_argument('--max-cond',type=int,default=3); ap.add_argument('--queries-per-size',type=int,default=3)
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df,audit=m2.load_data(args.data); names=list(df.columns); card=m2.cardinalities(names)
    print('[1/4] Repairing paired statistical analysis',flush=True)
    stats=paired_tests(Path(args.m3_results)/'repeat_level_metrics.csv',out)
    print('[2/4] Full-range exact inference validation',flush=True)
    inf=inference_validation(df,names,card,args.alpha,args.max_cond,out,args.seed,args.queries_per_size)
    print('[3/3] Supplementary 80/20 holdout: joint NLL and missing-evidence evaluation',flush=True)
    nll,miss=single_holdout_bn_extensions(df,names,card,args.alpha,args.max_cond,args.seed,out)
    meta={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'seed':args.seed,'holdout':'stratified 80/20 single split','alpha_pc':args.alpha,'max_conditioning_set_size':args.max_cond,'inference_queries_per_evidence_size':args.queries_per_size,'dataset_audit':audit,'notes':['Inference correctness validation uses a full-data fitted BN solely for algorithmic equivalence checking; it is not predictive test evaluation.','Joint NLL and missing-evidence evaluation use one stratified 80/20 holdout and are supplementary, not repeated-CV estimates.','Missing-evidence evaluation compares exact BN marginalization with train-statistic imputation on identical test masks.','Paired tests are exploratory because repeated CV runs reuse observations.']}
    (out/'m4_extension_config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('\nM4 extensions completed:',out.resolve())
    print('Inference max error:',inf.max_absolute_error.max())
    print('Statistical rows:',len(stats),' | Joint-NLL rows:',len(nll),' | Missing-evidence rows:',len(miss))

if __name__=='__main__': main()
