"""Check invariants needed for statistical interpretation, without further inference."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from analyze import auc_ci
q=json.loads(Path('results/questions.json').read_text())
assert len(q)==1600 and len({r['question'].strip() for r in q})==1600
assert len({r['qid'] for r in q})==1600
assert all(sorted(r['base_order'])==list(range(4)) for r in q)
# Compare weighted rank computation (including ties) with the library metric.
rng=np.random.default_rng(17)
for _ in range(30):
    y=rng.integers(0,2,80);s=rng.integers(0,6,80);w=rng.integers(1,6,80)
    o=np.argsort(s);ss=s[o];yy=y[o];ww=w[o]
    ix=np.r_[0,np.where(np.diff(ss)!=0)[0]+1]
    wp=np.add.reduceat(ww*yy,ix);wn=np.add.reduceat(ww*(1-yy),ix)
    rank_auc=(wp*(np.cumsum(wn)-wn/2)).sum()/(wp.sum()*wn.sum())
    assert abs(rank_auc-roc_auc_score(y,s,sample_weight=w))<1e-12
records=[r for f in sorted(Path('results/raw').glob('batch_*.json')) for r in json.loads(f.read_text())]
if records:
    assert len(records)==14400
    assert len({(r['qid'],r['condition']) for r in records})==14400
    for r in records:
        assert r['correct']==(r['pred']==r['gold'])
        assert r['pred']==r['order'][r['pred_choice']]
        assert abs(sum(r['probs'])-1)<1e-5
    cv=json.loads(Path('results/raw/cache_verification.json').read_text())
    assert cv['min_cosine']>.995
    for f in Path('results/raw').glob('batch_*.npz'):
        a=np.load(f)
        assert a['prompt'].shape==a['answer'].shape
        assert a['prompt'].shape[1:]==(4,3584)
        assert np.isfinite(a['prompt']).all() and np.isfinite(a['answer']).all()
Path('results/validation.json').write_text(json.dumps(dict(unique_questions=len(q),records_checked=len(records),weighted_auc_tests=30,passed=True),indent=2))
print('Validation passed:',len(q),'questions;',len(records),'responses')
