import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / 'data'
BASE = Path(__file__).resolve().parent
OUT_AUDIT = BASE / 'pubchem_source_audit.csv'
OUT_100 = BASE / 'pubchem_compounds_with_citation.csv'
USER_AGENT = 'SSAFY-AI-Core-AqSolDB-educational-project/1.0'
SAMPLE_SIZE = 300
TARGET = 100
BATCH_SIZE = 10


def request_json(url, attempts=4):
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            body = exc.read(300).decode('utf-8', 'replace')
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
                time.sleep((attempt + 1) * 2)
                continue
            return None, f'HTTP {exc.code}: {body[:120]}'
        except Exception as exc:
            if attempt + 1 < attempts:
                time.sleep((attempt + 1) * 2)
                continue
            return None, f'{type(exc).__name__}: {exc}'
    return None, 'request failed'


def read_rows():
    rows = []
    for split, filename in [('train', 'train.csv'), ('test', 'test.csv')]:
        with (DATA_DIR / filename).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                row['_split'] = split
                rows.append(row)
    return rows


def get_sample(rows):
    # Evenly spaced deterministic sample across both provided files.
    return [rows[round(i * (len(rows) - 1) / (SAMPLE_SIZE - 1))] for i in range(SAMPLE_SIZE)]


def map_keys_to_cids(sample):
    mapping = {}
    errors = {}
    for start in range(0, len(sample), BATCH_SIZE):
        batch = sample[start:start + BATCH_SIZE]
        keys = [row['InChIKey'] for row in batch]
        encoded = ','.join(urllib.parse.quote(key, safe='') for key in keys)
        url = ('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/'
               f'{encoded}/property/Title,InChIKey/JSON')
        data = request_json(url)
        time.sleep(0.3)
        if isinstance(data, tuple):
            for key in keys:
                errors[key] = data[1]
            continue
        returned = data.get('PropertyTable', {}).get('Properties', [])
        for prop in returned:
            key = prop.get('InChIKey')
            if key and key in keys and key not in mapping:
                mapping[key] = (prop.get('CID'), prop.get('Title', ''))
        for key in keys:
            if key not in mapping and key not in errors:
                errors[key] = 'No PubChem CID matched this exact InChIKey'
    return mapping, errors


def collect_strings(sections):
    found = []
    for section in sections or []:
        for info in section.get('Information', []):
            values = info.get('Value', {}).get('StringWithMarkup', [])
            text = ' '.join(value.get('String', '').strip() for value in values
                            if value.get('String', '').strip())
            if text:
                refs = info.get('Reference', [])
                extended = info.get('ExtendedReference', [])
                citations = []
                for ref in refs:
                    if isinstance(ref, str) and ref.strip():
                        citations.append(ref.strip())
                for ref in extended:
                    citation = ref.get('Citation') or ref.get('SourceName')
                    if citation and citation.strip():
                        citations.append(citation.strip())
                found.append((section.get('TOCHeading', ''), info.get('Description', ''),
                              text, ' | '.join(dict.fromkeys(citations))))
        found.extend(collect_strings(section.get('Section', [])))
    return found


def check_candidate(row, cid, title):
    record_url = f'https://pubchem.ncbi.nlm.nih.gov/compound/{cid}' if cid else ''
    result = {
        'split': row['_split'], 'AqSolDB_ID': row.get('ID', ''),
        'Name': row.get('Name', ''), 'InChIKey': row.get('InChIKey', ''),
        'PubChem_CID': cid or '', 'PubChem_Title': title or '',
        'PubChem_URL': record_url, 'heading': '', 'annotation_type': '',
        'annotation_text': '', 'source_citation': '',
        'reference_verified': 'no', 'rights_status':
        'annotation-level reuse terms not confirmed; review source and terms before redistribution',
        'audit_status': 'no CID' if not cid else 'pending'
    }
    if not cid:
        return result
    url = (f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON'
           '?heading=Record%20Description')
    data = request_json(url)
    time.sleep(0.35)
    if isinstance(data, tuple):
        result['audit_status'] = data[1]
        return result
    record = data.get('Record', {})
    entries = collect_strings(record.get('Section', []))
    if not entries:
        result['audit_status'] = 'no Record Description text'
        return result
    # Prefer descriptions that carry an explicit citation; keep the first one for the sample index.
    entry = next((item for item in entries if item[3]), entries[0])
    result['heading'], result['annotation_type'], result['annotation_text'], result['source_citation'] = entry
    if entry[3]:
        result['reference_verified'] = 'yes'
        result['audit_status'] = 'text and citation'
    else:
        result['audit_status'] = 'text without citation'
    return result


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = read_rows()
    sample = get_sample(rows)
    mapping, lookup_errors = map_keys_to_cids(sample)
    audited = []
    for i, row in enumerate(sample, start=1):
        cid, title = mapping.get(row['InChIKey'], (None, ''))
        result = check_candidate(row, cid, title)
        if not cid and row['InChIKey'] in lookup_errors:
            result['audit_status'] = lookup_errors[row['InChIKey']]
        audited.append(result)
        if i % 25 == 0:
            write_csv(OUT_AUDIT, audited)
            eligible = sum(x['reference_verified'] == 'yes' for x in audited)
            print(f'checked {i}/{SAMPLE_SIZE}; exact CID {sum(bool(x["PubChem_CID"]) for x in audited)}; text+citation {eligible}', flush=True)
    eligible = [row for row in audited if row['reference_verified'] == 'yes']
    write_csv(OUT_AUDIT, audited)
    write_csv(OUT_100, eligible[:TARGET])
    print(f'FINAL sample={len(audited)} exact CID={sum(bool(x["PubChem_CID"]) for x in audited)} '
          f'text+citation={len(eligible)} selected={min(TARGET, len(eligible))}', flush=True)
    print(f'audit={OUT_AUDIT}', flush=True)
    print(f'selected={OUT_100}', flush=True)

if __name__ == '__main__':
    main()

