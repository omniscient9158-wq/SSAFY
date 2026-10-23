from pathlib import Path
import json,csv,re,hashlib
R=Path('.');pages=json.loads((R/'research/provenance/description_review/original_pages.json').read_text(encoding='utf-8'));snap=json.loads((R/'research/provenance/description_review_records.json').read_text(encoding='utf-8'));rs={r['compound_id']:r for r in snap['records']}
train={r['ID']:r for r in csv.DictReader((R/'data/train.csv').open(encoding='utf-8-sig'))}
choices={'E-793':('20494','65-45-2','less than 1 mg/mL at 68°F'),'F-67':('20886','92-84-2','less than 1 mg/mL at 68°F'),'C-137':('21128','99-94-5','less than 1 mg/mL at 66°F'),'A-2073':('20154','80-07-9','less than 1 mg/mL at 68°F'),'B-1465':('16117','684-93-5','less than 1 mg/mL at 64°F'),'B-171':('3987','75-04-7','Very soluble'),'A-3999':('2906','106-48-9','10 to 50 mg/mL at 59°F'),'A-4301':('16062','62-55-5','5 to 10 mg/mL at 68°F')}
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ap=R/'data/approved_external_sources.csv';approved=read(ap);before=len(approved);decisions=[]
terms='https://m.cameochemicals.noaa.gov/help/reference/terms_and_conditions.htm; https://ntp.niehs.nih.gov/support/policies'
ntp='National Toxicology Program Chemical Repository Database (NTP, 1992), reproduced in NOAA CAMEO Chemicals.'
for cid,(sourceid,cas,value) in choices.items():
 record=rs[cid];row=train[cid];url=f'https://cameochemicals.noaa.gov/chemical/{sourceid}';page=pages[url];text=page['text'];ann=next(a for a in record['annotations'] if a['source_url']==url)
 assert record['identity_matches'] and row['InChIKey']==record['property_inchikey'] and cas in text
 raw='\n'.join(text[text.index('Water Solubility:'):].splitlines()[:(3 if value=='Very soluble' else 4)])
 assert re.sub(r'\s+',' ',raw)==f'Water Solubility: {value} (NTP, 1992)'
 docid=f'NOAA-CAMEO-CH{sourceid}-WATER-SOLUBILITY'
 match=f'Local Train InChIKey matches cached PubChem CID {record["pubchem_cid"]}; PubChem CAMEO annotation links CH{sourceid}; source name and CAS {cas} cross-checked. Review restricted to this compound and field.'
 excerpt=f'Water Solubility: {value} (NTP, 1992).'
 if not any(a['document_id']==docid for a in approved):
  approved.append(dict(zip(approved[0].keys(),[docid,cid,row['InChIKey'],record['pubchem_cid'],match,'Physical Properties / Water Solubility',excerpt,'NOAA CAMEO Chemicals',ntp,url,'NTP-attributed water-solubility fact approved for this educational index. CAMEO named third-party restricted fields excluded; NTP source attribution retained. Review does not grant a blanket license to other CAMEO fields.',terms,'2026-10-23','yes','Only water-solubility field and its original qualifier, unit and temperature. No safety, handling, health guidance, CAS-derived fields, precise LogS conversion or unstated measurement conditions. Original page snapshot 2026-10-22; field reviewed 2026-10-23.'])))
 decisions.append({'AqSolDB_ID':cid,'document_id':docid,'reviewed_on':'2026-10-23','source_snapshot_date':page['retrieved_on'],'source_url':url,'pubchem_cid':record['pubchem_cid'],'inchikey':row['InChIKey'],'identity_review':match,'approved_excerpt':excerpt,'source_text_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),'rights_basis_urls':terms,'outcome':'approved_bounded_water_solubility_excerpt','index_eligible':'yes'})
write(ap,approved)
mp=R/'data/document_discovery_manifest.csv';manifest=read(mp)
for row in manifest:
 if row['AqSolDB_ID'] in choices:
  a=next(a for a in approved if a['AqSolDB_ID']==row['AqSolDB_ID']);row.update({'document_url':a['source_url'],'document_heading':a['heading'],'discovery_evidence':'approved_external_sources.csv; preserved 2026-10-22 source snapshot','review_status':'reviewed_excerpt_available','description_character_count':str(len(a['approved_excerpt'])),'description_sha256':hashlib.sha256(a['approved_excerpt'].encode('utf-8')).hexdigest()})
assert len(manifest)==100
write(mp,manifest);write(R/'docs/evidence/document_review_approvals_20261023.csv',decisions)
print('Approved new bounded excerpts:',len(approved)-before,'total',len(approved),'compounds',len(set(a['AqSolDB_ID'] for a in approved)))
