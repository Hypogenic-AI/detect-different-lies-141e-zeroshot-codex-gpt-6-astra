"""Archive metadata from primary sources; no generated references."""
import requests,json,time
from pathlib import Path
from bs4 import BeautifulSoup
ids=['2503.03750','2511.16035','2502.03407','2602.01425','2603.10003','2609.00180','2607.20479','2510.09033','2606.12618','2009.03300','2304.13734','2412.15115']
Path('data/literature').mkdir(parents=True,exist_ok=True)
records=[]
for id in ids:
    url='https://arxiv.org/abs/'+id
    r=requests.get(url,timeout=60);s=BeautifulSoup(r.text,'html.parser')
    rec=dict(id=id,url=url,status=r.status_code)
    for key,css in [('title','h1.title'),('authors','div.authors'),('abstract','blockquote.abstract')]:
        el=s.select_one(css);rec[key]=el.get_text(' ',strip=True) if el else None
    records.append(rec)
    Path(f'data/literature/{id}.html').write_text(r.text)
Path('results/literature.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
