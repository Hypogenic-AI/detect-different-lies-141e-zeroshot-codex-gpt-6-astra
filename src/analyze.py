"""Question-disjoint probes, cluster bootstrap intervals, and paper artifacts."""
import os,json,argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS','4');os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results'); FIG=Path('paper_draft/figures'); FIG.mkdir(parents=True,exist_ok=True)

def auc_ci(y,s,q,seed=1234,B=2000,return_samples=False):
    y=np.asarray(y,dtype=int);s=np.asarray(s);q=np.asarray(q)
    if len(set(y))<2:return [None,None,None]
    auc=float(roc_auc_score(y,s))
    uq,qi=np.unique(q,return_inverse=True);n=len(uq)
    rng=np.random.default_rng(seed);w=rng.multinomial(n,np.full(n,1/n),size=B)[:,qi]
    order=np.argsort(s);ys=y[order];ss=s[order];w=w[:,order]
    ix=np.r_[0,np.where(np.diff(ss)!=0)[0]+1]
    wp=np.add.reduceat(w*ys,ix,axis=1);wn=np.add.reduceat(w*(1-ys),ix,axis=1)
    denom=wp.sum(1)*wn.sum(1);valid=denom>0
    v=(wp*(np.cumsum(wn,axis=1)-wn/2)).sum(1)[valid]/denom[valid]
    lo,hi=np.quantile(v,[.025,.975]);return (auc,v) if return_samples else [auc,float(lo),float(hi)]

def load():
    rec=[];ph=[];ah=[]
    for file in sorted((OUT/'raw').glob('batch_*.json')):
        rr=json.loads(file.read_text());f=file.with_suffix('.npz');acts={k:v for k,v in np.load(f).items()} if f.exists() else None
        for i,r in enumerate(rr):
            if not r['condition'].startswith('belief'):
                assert acts is not None
                r['fidx']=len(ph);ph.append(acts['prompt'][i]);ah.append(acts['answer'][i])
            rec.append(r)
    df=pd.DataFrame(rec); assert len(df)==1600*9, len(df)
    assert not df.duplicated(['qid','condition']).any()
    beliefs=df[df.condition.str.startswith('belief')]
    knowledge={};kcount={}
    for q,g in beliefs.groupby('qid'):
        assert len(g)==4
        kcount[q]=int(g.correct.sum())
        knowledge[q]='K' if g.correct.all() else ('W' if g.pred.nunique()==1 and not g.correct.any() else 'U')
    df['knowledge']=df.qid.map(knowledge);df['kcount']=df.qid.map(kcount)
    df['false']=~df.correct;df['target_correct']=df.target==df.gold
    df['compliance']=df.pred==df.target
    df['entropy']=df.probs.apply(lambda p:float(-np.sum(np.array(p)*np.log(np.maximum(p,1e-30)))))
    df['uncertainty']=df.probs.apply(lambda p:1-max(p))
    df.drop(columns=['prompt','order','probs','choice_logits']).to_csv(OUT/'responses.csv',index=False)
    return df,np.stack(ph).astype('float32'),np.stack(ah).astype('float32')

def behavior(df):
    ev=df[~df.condition.str.startswith('belief')].copy();rows=[]
    for cond,g in ev.groupby('condition'):
        wrong=g[g.false]
        r=dict(condition=cond,n=len(g),correct=int(g.correct.sum()),false=int(g.false.sum()),compliance=float(g.compliance.mean()),unrestricted_choice=float(g.unrestricted_is_choice.mean()),choice_mass_mean=float(g.choice_mass.mean()))
        for k in ['K','U','W']:
            z=g[g.knowledge==k];r[k+'_n']=len(z);r[k+'_false']=int(z.false.sum());r[k+'_share_false']=float((wrong.knowledge==k).mean())
            # bootstrap proportion is binomial here, one row per question, report exact CI
            ci=binomtest(r[k+'_false'],len(wrong)).proportion_ci()
            r[k+'_share_ci']=[float(ci.low),float(ci.high)]
        conflict=g[(g.knowledge=='K') & ~g.target_correct]
        r['K_errors_selecting_target']=int(((g.knowledge=='K') & g.false & g.compliance).sum());r['K_conflict_n']=len(conflict);r['K_conflict_false']=int(conflict.false.sum());r['K_conflict_correct']=int(conflict.correct.sum())
        rows.append(r)
    (OUT/'behavior.json').write_text(json.dumps(rows,indent=2))
    pd.DataFrame(rows).to_csv(OUT/'behavior.csv',index=False)
    kn=df[df.condition=='neutral'].groupby(['split','knowledge']).size().unstack(fill_value=0)
    kn.to_csv(OUT/'knowledge_counts.csv')
    # Matched pressure minus neutral error rate, with paired exact McNemar/binomial test.
    paired=[]
    for k in ['all','K','U','W']:
        base=ev if k=='all' else ev[ev.knowledge==k]
        piv=base.pivot(index='qid',columns='condition',values='false').astype(int)
        for c in ['score','social']:
            d=(piv[c]-piv.neutral).to_numpy(); rng=np.random.default_rng(33)
            bs=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
            up=int((d==1).sum());down=int((d==-1).sum())
            paired.append(dict(knowledge=k,condition=c,n=len(d),increase=float(d.mean()),ci=np.quantile(bs,[.025,.975]).tolist(),up=up,down=down,p=binomtest(up,up+down).pvalue if up+down else 1.))
    (OUT/'paired_effects.json').write_text(json.dumps(paired,indent=2))
    counts=ev[ev.condition.isin(['neutral','score','social'])].groupby(['condition','knowledge']).false.sum().unstack().loc[['neutral','score','social'],['K','U','W']]
    ax=counts.plot.bar(stacked=True,color=['#b85c38','#387ca3','#8a8491'],figsize=(6,3.3),rot=0)
    ax.set_ylabel('Number of incorrect answers');ax.set_xlabel('Reporting condition');ax.legend(['Stable-correct (K)','Inconsistent (U)','Stable-wrong (W)'],fontsize=8)
    plt.tight_layout();plt.savefig(FIG/'composition.pdf');plt.close()
    return rows

def mask_task(e,kind,condition='score',relaxed=False):
    K=e.kcount>=3 if relaxed else e.knowledge=='K'; U=(e.knowledge=='U') & ~K
    f=e.false;c=e.correct;cond=e.condition
    if kind=='instruction':pos=K & (cond=='lie') & f;neg=K & (cond=='honest') & c
    elif kind=='knowledge':pos=K & (cond=='neutral');neg=U & (cond=='neutral')
    elif kind=='error':pos=(cond=='neutral') & f;neg=(cond=='neutral') & c
    elif kind=='pressure':pos=K & (cond==condition) & f;neg=K & (cond==condition) & c
    elif kind=='cross':pos=K & (cond==condition) & f;neg=U & (cond=='neutral') & f
    elif kind=='within':pos=K & (cond==condition) & f;neg=U & (cond==condition) & f
    elif kind=='stable_wrong':pos=K & (cond==condition) & f;neg=(e.knowledge=='W') & (cond==condition) & f
    elif kind=='conflict':pos=K & (cond==condition) & f & ~e.target_correct;neg=K & (cond==condition) & c & ~e.target_correct
    else:raise ValueError(kind)
    return (pos|neg).to_numpy(),pos.to_numpy().astype(int)

def probes(df,ph,ah):
    e=df[df.fidx.notna()].copy().reset_index(drop=True);idx=e.fidx.astype(int).to_numpy();spl=e.split.to_numpy();q=e.qid.to_numpy()
    results=[];score_rows=[];sizes=[];primary_scores={}
    (OUT/'probes').mkdir(exist_ok=True)
    conditions=['score','social'];tasks=[('instruction','score'),('error','score'),('knowledge','score')]+[(k,c) for c in conditions for k in ['pressure','cross','within','stable_wrong','conflict']]
    # Primary and layer robustness fits; fixed C, no held-out metric used for selection.
    for pos,h in [('prompt',ph),('answer',ah)]:
        for li,layer in enumerate([7,14,21,28]):
            X=h[idx,li]
            for trainkind in ['instruction','error','pressure','cross','within','knowledge']:
                m,y=mask_task(e,trainkind);tr=m & (spl=='train');va=m & (spl=='val')
                if len(np.unique(y[tr]))<2:continue
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,solver='lbfgs'))
                clf.fit(X[tr],y[tr]);score=clf.decision_function(X)
                np.savez_compressed(OUT/'probes'/f'{pos}_{layer}_{trainkind}.npz',mean=clf[0].mean_,scale=clf[0].scale_,coef=clf[1].coef_,intercept=clf[1].intercept_)
                if layer==14:primary_scores[(pos,trainkind)]=score
                threshold=float(np.quantile(score[va & (y==0)],.9))
                sizes.append(dict(position=pos,layer=layer,probe=trainkind,train_n=int(tr.sum()),train_pos=int(y[tr].sum()),val_n=int(va.sum()),threshold=threshold,iterations=int(clf[-1].n_iter_[0])))
                for evalkind,condition in tasks:
                    m2,y2=mask_task(e,evalkind,condition);te=m2 & (spl=='test')
                    # intervals for primary layer; all point estimates retained elsewhere
                    ci=auc_ci(y2[te],score[te],q[te],B=2000 if layer==14 else 400)
                    results.append(dict(position=pos,layer=layer,probe=trainkind,task=evalkind,condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
                if layer==14:
                    for i in range(len(e)):
                        if spl[i]=='test':score_rows.append(dict(qid=int(q[i]),condition=e.condition[i],knowledge=e.knowledge[i],correct=bool(e.correct[i]),probe=trainkind,position=pos,score=float(score[i]),flag=bool(score[i]>=threshold)))
                    # relaxed knowledge-definition sensitivity, same model coefficients
                    for evalkind in ['cross','within']:
                        for condition in conditions:
                            m2,y2=mask_task(e,evalkind,condition,relaxed=True);te=m2 & (spl=='test');ci=auc_ci(y2[te],score[te],q[te])
                            results.append(dict(position=pos,layer=layer,probe=trainkind,task=evalkind+'_relaxed',condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
                print(pos,layer,trainkind,'done',flush=True)
    differences=[]
    for kind in ['instruction','error','pressure','cross','within','knowledge']:
        for task in ['cross','within']:
            for condition in conditions:
                m,y=mask_task(e,task,condition);te=m&(spl=='test')
                av,ab=auc_ci(y[te],primary_scores[('answer',kind)][te],q[te],return_samples=True)
                pv,pb=auc_ci(y[te],primary_scores[('prompt',kind)][te],q[te],return_samples=True)
                differences.append(dict(probe=kind,task=task,condition=condition,answer_minus_prompt=av-pv,lo=float(np.quantile(ab-pb,.025)),hi=float(np.quantile(ab-pb,.975))))
    pd.DataFrame(differences).to_csv(OUT/'position_differences.csv',index=False)
    # Simple non-hidden-state controls, fit on train only.
    for name,X in [('answer_letter',np.eye(4)[e.pred_choice.to_numpy()]),('surface',np.column_stack([np.eye(4)[e.pred_choice.to_numpy()],e.n_tokens,e.entropy,e.uncertainty,e.choice_mass]))]:
        for trainkind in ['cross','within','pressure']:
            m,y=mask_task(e,trainkind);tr=m & (spl=='train')
            clf=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000)).fit(X[tr],y[tr]);scores=clf.decision_function(X)
            for evalkind,condition in tasks:
                m2,y2=mask_task(e,evalkind,condition);te=m2&(spl=='test');ci=auc_ci(y2[te],scores[te],q[te])
                results.append(dict(position=name,layer=0,probe=trainkind,task=evalkind,condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
    # A condition-only score exposes the cross-prompt label shortcut directly.
    for evalkind,condition in tasks:
        m2,y2=mask_task(e,evalkind,condition);te=m2&(spl=='test')
        ci=auc_ci(y2[te],(e.condition.to_numpy()!='neutral').astype(float)[te],q[te])
        results.append(dict(position='condition_only',layer=0,probe='condition',task=evalkind,condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
    # Question text alone tests dataset difficulty/topic separation between K and U.
    qr={r['qid']:r['question'] for r in json.loads((OUT/'questions.json').read_text())}
    for trainkind in ['cross','within']:
        m,y=mask_task(e,trainkind);tr=m&(spl=='train')
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000)
        Xtr=vec.fit_transform([qr[i] for i in q[tr]])
        clf=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000).fit(Xtr,y[tr])
        score=clf.decision_function(vec.transform([qr[i] for i in q]))
        for evalkind,condition in tasks:
            m2,y2=mask_task(e,evalkind,condition);te=m2&(spl=='test');ci=auc_ci(y2[te],score[te],q[te])
            results.append(dict(position='question_text',layer=0,probe=trainkind,task=evalkind,condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
    for score_name in ['entropy','uncertainty']:
        for evalkind,condition in tasks:
            m2,y2=mask_task(e,evalkind,condition);te=m2&(spl=='test');ci=auc_ci(y2[te],e[score_name].to_numpy()[te],q[te])
            results.append(dict(position='scalar',layer=0,probe=score_name,task=evalkind,condition=condition,n=int(te.sum()),n_pos=int(y2[te].sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
    pd.DataFrame(results).to_csv(OUT/'probe_metrics.csv',index=False);pd.DataFrame(sizes).to_csv(OUT/'probe_training.csv',index=False)
    sc=pd.DataFrame(score_rows);sc.to_csv(OUT/'test_scores.csv',index=False)
    flag=sc.groupby(['position','probe','condition','knowledge','correct']).agg(n=('flag','size'),flag_rate=('flag','mean'),mean_score=('score','mean')).reset_index()
    flag.to_csv(OUT/'flag_rates.csv',index=False)
    r=pd.DataFrame(results)
    fig,axes=plt.subplots(1,2,figsize=(8,3.4),sharey=True)
    for ax,task in zip(axes,['cross','within']):
        names=['instruction','error','pressure','cross','within'];xx=np.arange(len(names))
        for p,off,col in [('answer',-.13,'#b85c38'),('prompt',.13,'#387ca3')]:
            v=r[(r.layer==14)&(r.position==p)&(r.task==task)&(r.condition=='social')].set_index('probe').loc[names]
            ax.errorbar(xx+off,v.auc,yerr=[v.auc-v.lo,v.hi-v.auc],fmt='o',capsize=2,color=col,label=p)
        ax.axhline(.5,color='gray',ls='--',lw=1);ax.set_xticks(xx,names,rotation=35,ha='right');ax.set_title('Cross-prompt errors' if task=='cross' else 'Same-pressure errors');ax.set_ylim(0,1.04);ax.set_ylabel('Test AUROC (K error positive)')
    axes[0].legend(fontsize=8);plt.tight_layout();plt.savefig(FIG/'probes.pdf');plt.close()
    return r

def main():
    df,p,a=load();behavior(df);probes(df,p,a)
    print('Analysis complete',flush=True)
if __name__=='__main__':main()
