"""Audit stored candidate provenance without approving or indexing any text."""
from __future__ import annotations

import csv
import argparse
import json
import html
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def pending_candidates() -> list[dict[str, str]]:
    with (BASE / 'data/document_discovery_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row['review_status'] != 'reviewed_excerpt_available']


def annotation_rows(record: dict, references: dict, heading: str = '') -> list[dict]:
    result = []
    for section in record.get('Section', []):
        path = f"{heading} / {section.get('TOCHeading', '')}".strip(' /')
        for info in section.get('Information', []):
            text = ' '.join(value.get('String', '') for value in info.get('Value', {}).get('StringWithMarkup', []))
            ref = references.get(info.get('ReferenceNumber'), {})
            result.append({'heading': path, 'text': text, 'provider': ref.get('SourceName', ''),
                           'source_id': ref.get('SourceID', ''), 'source_url': ref.get('URL', ''),
                           'license_note': ref.get('LicenseNote', ''), 'license_url': ref.get('LicenseURL', ''),
                           'citations': info.get('Reference', [])})
        result.extend(annotation_rows(section, references, path))
    return result


def fetch_source_records() -> None:
    """Fetch provenance only; never call models or approve/index any material."""
    from discover_wikipedia_candidates import _get
    from requests import RequestException

    candidates = pending_candidates()
    cache = BASE / 'research/provenance/description_review'
    cache.mkdir(parents=True, exist_ok=True)
    property_path = cache / 'identity_properties.json'
    properties = json.loads(property_path.read_text(encoding='utf-8')) if property_path.exists() else {}
    missing = [row['PubChem_CID'] for row in candidates if row['PubChem_CID'] not in properties]
    for start in range(0, len(missing), 10):
        cids = ','.join(missing[start:start + 10])
        response = _get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/Title,InChIKey/JSON', timeout=20)
        response.raise_for_status()
        for value in response.json().get('PropertyTable', {}).get('Properties', []):
            properties[str(value['CID'])] = value
        property_path.write_text(json.dumps(properties, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Identity properties cached: {len(properties)}/{len(candidates)}', flush=True)
    output = []
    for number, item in enumerate(candidates, 1):
        cid = item['AqSolDB_ID']
        target = cache / f'{cid}.json'
        api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{item['PubChem_CID']}/JSON?heading=Record%20Description"
        try:
            if target.exists():
                record = json.loads(target.read_text(encoding='utf-8'))
            else:
                response = _get(api_url, timeout=20)
                response.raise_for_status()
                record = response.json()['Record']
                target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            references = {ref['ReferenceNumber']: ref for ref in record.get('Reference', [])}
            identity = properties.get(item['PubChem_CID'], {})
            output.append({'compound_id': cid, 'pubchem_cid': item['PubChem_CID'],
                           'pubchem_title': record.get('RecordTitle', ''),
                           'identity_matches': identity.get('InChIKey') == item['InChIKey'],
                           'property_inchikey': identity.get('InChIKey', ''), 'api_url': api_url,
                           'cache_file': target.relative_to(BASE).as_posix(),
                           'annotations': annotation_rows(record, references), 'error': ''})
        except (OSError, ValueError, KeyError, RuntimeError, RequestException) as error:
            output.append({'compound_id': cid, 'api_url': api_url, 'error': type(error).__name__})
        # Checkpoint each record so an interruption never discards the review inputs.
        result_path = BASE / 'research/provenance/description_review_records.json'
        result_path.write_text(json.dumps({'assembled_on': datetime.now(timezone.utc).isoformat(),
                                          'records': output}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Provenance {number}/{len(candidates)}: {cid}; annotations={len(output[-1].get("annotations", []))}; error={output[-1]["error"]}', flush=True)


def fetch_original_pages() -> None:
    """Archive official CAMEO source pages and rights policies for manual reading."""
    from discover_wikipedia_candidates import _get
    from requests import RequestException

    records = json.loads((BASE / 'research/provenance/description_review_records.json').read_text(encoding='utf-8'))['records']
    urls = {a['source_url'] for r in records for a in r.get('annotations', []) if a['provider'] == 'CAMEO Chemicals'}
    urls.update({'https://m.cameochemicals.noaa.gov/help/reference/terms_and_conditions.htm',
                 'https://ntp.niehs.nih.gov/support/policies',
                 'https://www.noaa.gov/office-education/outreach-communication/faq'})
    cache = BASE / 'research/provenance/description_review'
    target = cache / 'original_pages.json'
    pages = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    for number, url in enumerate(sorted(urls), 1):
        if url not in pages or pages[url].get('error'):
            try:
                response = _get(url, timeout=20)
                response.raise_for_status()
                markup = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', '', response.text, flags=re.S | re.I)
                readable = html.unescape(re.sub('<[^>]+>', '\n', markup))
                readable = '\n'.join(line.strip() for line in readable.splitlines() if line.strip())
                pages[url] = {'url': response.url, 'retrieved_on': datetime.now(timezone.utc).isoformat(),
                              'http_status': response.status_code, 'text': readable, 'error': ''}
            except (OSError, ValueError, RequestException) as error:
                pages[url] = {'error': type(error).__name__}
            target.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Original page {number}/{len(urls)}: {url}; error={pages[url]["error"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch-source-records', action='store_true', help='Fetch/cache PubChem provenance, without model calls or approvals')
    parser.add_argument('--fetch-original-pages', action='store_true', help='Fetch/cache official CAMEO pages and rights policies for manual review')
    args = parser.parse_args()
    if args.fetch_source_records:
        fetch_source_records()
    if args.fetch_original_pages:
        fetch_original_pages()
    with (BASE / 'data/document_discovery_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    with (BASE / 'research/provenance/pubchem_source_audit.csv').open(encoding='utf-8-sig', newline='') as stream:
        audit = {row['AqSolDB_ID']: row for row in csv.DictReader(stream)}
    records = []
    for item in manifest:
        if item['review_status'] == 'reviewed_excerpt_available':
            continue
        source = audit.get(item['AqSolDB_ID'], {})
        text = source.get('annotation_text', '').strip()
        citation = source.get('source_citation', '').strip()
        if item['AqSolDB_ID'] == 'D-1360':
            status = 'hold_identity_conflict'
            next_step = 'Stored hydroxy compound name conflicts with amino compound CID 6270; see docs/03_출처_검토_보고서.md.'
        elif item['AqSolDB_ID'] == 'B-1198':
            status = 'hold_citation_subject_conflict'
            next_step = 'Pentene description cites Pentane; verify the original article and revision before reuse.'
        elif not text or text.lower().startswith(('see ', 'structure ', 'rn given')):
            status = 'replace_non_substantive_description'
            next_step = 'Find a substantive compound-specific property description.'
        elif any(term in text.lower() for term in ('safe in', 'safe as', 'toxicity', 'clinical trials', 'storage of', 'sensitizing')):
            status = 'replace_out_of_scope_description'
            next_step = 'Find physical-property text rather than health or handling guidance.'
        elif not citation:
            status = 'hold_missing_original_attribution'
            next_step = 'Resolve annotation provider and original source before checking reuse terms.'
        else:
            status = 'hold_original_source_and_terms_review'
            next_step = 'Inspect original source, compound identity, excerpt relevance and reuse conditions.'
        records.append({
            'AqSolDB_ID': item['AqSolDB_ID'], 'document_url': item['document_url'],
            'identity_matches_stored_audit': str(item['InChIKey'] == source.get('InChIKey')),
            'description_characters': len(text), 'has_original_citation': bool(citation),
            'triage_status': status, 'next_step': next_step,
            'index_eligible': 'no',
        })
    target = BASE / 'docs/evidence/document_candidate_triage.csv'
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(dict(Counter(row['triage_status'] for row in records)))
    print(f'Triaged {len(records)} candidates; newly approved: 0; network/API calls: 0')


if __name__ == '__main__':
    main()
