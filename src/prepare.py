"""Download immutable model/dataset snapshots and draw a fixed MMLU sample."""
import os,json,random,hashlib
from pathlib import Path
os.environ.setdefault('HF_HOME', str(Path('.cache/huggingface').resolve()))
from huggingface_hub import HfApi,snapshot_download
from datasets import load_dataset
Path('results').mkdir(exist_ok=True)
api=HfApi()
model='Qwen/Qwen2.5-7B-Instruct'; dataset='cais/mmlu'
mr='a09a35458c702b33eeacc393d103063234e8bc28'; dr='c30699e8356da336a370243923dbaf21066bb9fe'
print('Revisions',mr,dr,flush=True)
snapshot_download(model,revision=mr,local_dir='models/qwen',allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
ds=load_dataset(dataset,'all',split='test',revision=dr)
rng=random.Random(20261002)
ids=rng.sample(range(len(ds)),1600)
# Remove exact duplicate stems before any full-study inference or split fitting.
seen=set(); replacements=[]
for j,idx in enumerate(ids):
    while ds[idx]['question'].strip() in seen:
        old=idx; idx=rng.choice([k for k in range(len(ds)) if k not in ids]); ids[j]=idx
        replacements.append({'position':j,'old_source_index':old,'new_source_index':idx})
    seen.add(ds[idx]['question'].strip())
Path('results/deduplication.json').write_text(json.dumps(replacements,indent=2))
rows=[]
for i,idx in enumerate(ids):
    r=ds[idx]; base=list(range(4)); rng.shuffle(base)
    r.update(qid=i,source_index=idx,base_order=base,target=rng.randrange(4),split='train' if i<800 else ('val' if i<1100 else 'test'))
    rows.append(r)
Path('results/questions.json').write_text(json.dumps(rows,indent=2))
Path('results/provenance.json').write_text(json.dumps(dict(model=model,model_revision=mr,dataset=dataset,dataset_revision=dr,seed=20261002,n=len(rows),dataset_size=len(ds),question_sha256=hashlib.sha256(Path('results/questions.json').read_bytes()).hexdigest()),indent=2))
print('Prepared',len(rows),flush=True)
