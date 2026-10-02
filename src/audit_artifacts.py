"""Final artifact integrity, PDF, provenance, and accidental-secret checks."""
import hashlib,json,os
from pathlib import Path
import pymupdf
root=Path('.')
prov=json.loads(Path('results/provenance.json').read_text())
assert hashlib.sha256(Path('results/questions.json').read_bytes()).hexdigest()==prov['question_sha256']
pdf=pymupdf.open('paper_draft/main.pdf');txt='\n'.join(p.get_text() for p in pdf)
assert '[?]' not in txt and 'PLACEHOLDER' not in txt and len(txt)>20000
assert Path('README.md').stat().st_size>1000
files=[]
for folder in ['src','results','paper_draft']:
    for p in Path(folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:continue
        if p.name in ['artifact_manifest.json','artifact_audit.json']:continue
        if p.suffix in ['.aux','.log','.out','.bbl','.blg']:continue
        files.append(p)
files += [Path('README.md'),Path('requirements.txt')]
credentials=[v.encode() for k in ['OPENAI_API_KEY','OPENROUTER_KEY','HF_TOKEN'] if (v:=os.environ.get(k)) and len(v)>8]
manifest=[]
for p in sorted(files):
    data=p.read_bytes()
    if p.suffix in ['.py','.json','.csv','.md','.tex','.bib','.txt']:
        assert not any(key in data for key in credentials), f'Credential substring found in {p}'
    manifest.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
Path('results/artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
Path('results/artifact_audit.json').write_text(json.dumps(dict(passed=True,pdf_pages=len(pdf),pdf_text_characters=len(txt),manifest_files=len(manifest),question_provenance_hash_matches=True,credential_substrings_found=False),indent=2))
print('Final artifact audit passed:',len(pdf),'PDF pages;',len(manifest),'hashed files')
