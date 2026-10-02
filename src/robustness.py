"""Post-hoc target-adherence and stricter-belief checks; no probe retraining."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from analyze import auc_ci
r=pd.read_csv('results/responses.csv');s=pd.read_csv('results/test_scores.csv')
meta=r[r.split=='test'].set_index(['qid','condition'])
neut=r[r.condition=='neutral'].set_index('qid').correct
rows=[]
for condition in ['score','social']:
    base=r[(r.split=='test')&(r.condition==condition)&r.false&r.knowledge.isin(['K','U'])].copy()
    y=(base.knowledge=='K').astype(int).to_numpy()
    ci=auc_ci(y,base.compliance.astype(float).to_numpy(),base.qid.to_numpy())
    rows.append(dict(check='adherence_only',condition=condition,probe='compliance',position='scalar',n=len(base),n_pos=int(y.sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
    for check in ['both_adherent','conflicting_target','neutral_verified']:
        v=base.copy()
        if check=='both_adherent':v=v[v.compliance]
        elif check=='conflicting_target':v=v[~v.target_correct]
        elif check=='neutral_verified':v=v[(v.knowledge!='K')|v.qid.map(neut)]
        for pos in ['answer','prompt']:
            for probe in s.probe.unique():
                z=s[(s.position==pos)&(s.probe==probe)&(s.condition==condition)].set_index('qid').loc[v.qid]
                y=(v.knowledge=='K').astype(int).to_numpy();ci=auc_ci(y,z.score.to_numpy(),v.qid.to_numpy())
                rows.append(dict(check=check,condition=condition,probe=probe,position=pos,n=len(v),n_pos=int(y.sum()),auc=ci[0],lo=ci[1],hi=ci[2]))
pd.DataFrame(rows).to_csv('results/robustness.csv',index=False)
print(pd.DataFrame(rows).query("condition=='social' and position!='prompt'").to_string(index=False))
