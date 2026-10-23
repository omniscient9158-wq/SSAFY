from pathlib import Path
import csv,json,re,hashlib
R=Path('.');source=json.loads((R/'research/provenance/description_review_records.json').read_text(encoding='utf-8'));pages=json.loads((R/'research/provenance/description_review/original_pages.json').read_text(encoding='utf-8'));approved={r['AqSolDB_ID'] for r in csv.DictReader((R/'docs/evidence/document_review_approvals_20261023.csv').open(encoding='utf-8-sig'))};rows=[]
for record in source['records']:
 cid=record['compound_id'];originals=[]
 for a in record.get('annotations',[]):
  p=pages.get(a.get('source_url'),{})
  if a['provider']=='CAMEO Chemicals' and p.get('http_status')==200 and not p.get('error'):
   text=p['text'];i=text.find('Water Solubility:');raw=text[i:i+200].split('Ionization')[0].split('IDLH')[0] if i>=0 else '';originals.append((a['source_url'],raw))
 if cid in approved:outcome='approved_bounded_excerpt';reason='NTP-attributed water-solubility field approved; identity, scope and attribution recorded.'
 elif cid in ['D-1360','B-1198','A-1068']:outcome='hold_identity_or_citation_conflict';reason={'D-1360':'Hydroxy local name conflicts with amino source identity; InChIKey agreement does not resolve the name/structure conflict.','B-1198':'Prior Pentane citation conflict preserved; cached CAMEO records do not supply a water-solubility fact.','A-1068':'Local mixture label Fusel oil conflicts with single-compound Isoamyl Alcohol source.'}[cid]
 elif any('NIOSH' in raw and 'data unavailable' not in raw for _,raw in originals):outcome='hold_alternative_source_terms';reason='Water-solubility field is attributed to NIOSH; the NTP excerpt decision is not applied to a different provider.'
 elif originals:outcome='exclude_cached_source_no_eligible_water_solubility';reason='Inspected cached CAMEO water-solubility field does not contain an eligible NTP-attributed value; no number inferred from hazards or appearance.'
 else:outcome='hold_original_definition_verification';reason='PubChem provider annotations cached and identity matched, but an original compound-specific property/definition and its usage conditions require verification; no new content approved.'
 rows.append({'AqSolDB_ID':cid,'pubchem_title':record['pubchem_title'],'reviewed_on':'2026-10-23','identity_matches_cached_properties':str(record['identity_matches']),'cached_original_urls':'; '.join(u for u,_ in originals),'inspected_water_solubility_fields':' | '.join(re.sub(r'\s+',' ',v).strip() for _,v in originals),'disposition':outcome,'decision_reason':reason,'index_eligible':'yes' if cid in approved else 'no'})
p=R/'docs/evidence/document_review_dispositions_20261023.csv'
with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
from collections import Counter
print(dict(Counter(r['disposition'] for r in rows)))
