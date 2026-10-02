"""Select short illustrative examples by ascending question ID, not detector score."""
import json
from pathlib import Path
import pandas as pd
r=pd.read_csv('results/responses.csv');qs={x['qid']:x for x in json.loads(Path('results/questions.json').read_text())}
p=r.pivot(index='qid',columns='condition',values='pred');k=r[r.condition=='neutral'].set_index('qid').knowledge
examples=[]
for name in ['pressure_contradiction','neutral_inconsistent_error','stable_wrong_error']:
    for q in sorted(qs):
        row=qs[q];gold=row['answer']
        if len(row['question'])>180:continue
        ok=(k[q]=='K' and p.loc[q,'neutral']==gold and p.loc[q,'social']!=gold) if name=='pressure_contradiction' else ((k[q]=='U' and p.loc[q,'neutral']!=gold) if name=='neutral_inconsistent_error' else k[q]=='W' and p.loc[q,'neutral']!=gold)
        if ok:
            examples.append(dict(category=name,question=row,neutral_answer=row['choices'][p.loc[q,'neutral']],social_answer=row['choices'][p.loc[q,'social']],gold_answer=row['choices'][gold],preferred_answer=row['choices'][row['target']],belief_answers=[row['choices'][p.loc[q,f'belief{i}']] for i in range(4)]));break
Path('results/examples.json').write_text(json.dumps(examples,indent=2))
print(json.dumps(examples,indent=2))
