"""Audit exact Train InChIKey links, names and Wikipedia availability (no GMS).

Successful batches are cached, including negative results. Failed batches stay
unknown rather than being counted as absent. Multiple QIDs are kept ambiguous.
This collects CC0 structured metadata, not Wikipedia article text or approval.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import compound_data
from discover_wikipedia_candidates import _get, _request_headers

ENDPOINT = "https://query.wikidata.org/sparql"
CACHE_DIR = compound_data.PROJECT_DIR / "research/provenance/wikidata_batches"
LINKS_PATH = compound_data.DATA_DIR / "wikidata_compound_links.jsonl"
REPORT_PATH = compound_data.PROJECT_DIR / "docs/evidence/wikidata_coverage_summary.json"
KEY_PATTERN = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]\Z")


def make_query(keys: list[str]) -> str:
    if not keys or any(not KEY_PATTERN.fullmatch(key) for key in keys):
        raise ValueError("Expected nonempty exact InChIKey values.")
    values = " ".join(json.dumps(key) for key in keys)
    return f'''PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
SELECT ?ik ?item ?en ?ko ?alias ?en_wiki ?ko_wiki WHERE {{
 VALUES ?ik {{ {values} }}
 ?item wdt:P235 ?ik .
 OPTIONAL {{ ?item rdfs:label ?en . FILTER(LANG(?en) = "en") }}
 OPTIONAL {{ ?item rdfs:label ?ko . FILTER(LANG(?ko) = "ko") }}
 OPTIONAL {{ ?item skos:altLabel ?alias . FILTER(LANG(?alias) IN ("en", "ko")) }}
 OPTIONAL {{ ?en_wiki schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> . }}
 OPTIONAL {{ ?ko_wiki schema:about ?item ; schema:isPartOf <https://ko.wikipedia.org/> . }}
}}'''


def merge_bindings(keys: list[str], bindings: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, dict[str, dict]] = {key: {} for key in keys}
    for binding in bindings:
        key = binding.get("ik", {}).get("value", "")
        item_url = binding.get("item", {}).get("value", "")
        qid = item_url.rsplit("/", 1)[-1]
        if key not in grouped or not re.fullmatch(r"Q[1-9][0-9]*", qid):
            raise ValueError("Unexpected identity in Wikidata response.")
        item = grouped[key].setdefault(qid, {
            "qid": qid, "source_url": f"https://www.wikidata.org/wiki/{qid}",
            "label_en": "", "label_ko": "", "aliases_en": [], "aliases_ko": [],
            "en_wikipedia": "", "ko_wikipedia": "",
        })
        for field, source in (("label_en", "en"), ("label_ko", "ko"), ("en_wikipedia", "en_wiki"), ("ko_wikipedia", "ko_wiki")):
            value = binding.get(source, {}).get("value", "")
            if value:
                if field.endswith("wikipedia"):
                    expected = "en" if field.startswith("en") else "ko"
                    if not value.startswith(f"https://{expected}.wikipedia.org/wiki/"):
                        raise ValueError("Unexpected Wikipedia URL.")
                if item[field] and item[field] != value:
                    raise ValueError("Conflicting entity fields in response.")
                item[field] = value
        alias = binding.get("alias", {})
        language = alias.get("xml:lang")
        if language in ("en", "ko") and alias.get("value"):
            item[f"aliases_{language}"].append(alias["value"])
    for entities in grouped.values():
        for item in entities.values():
            for language in ("en", "ko"):
                item[f"aliases_{language}"] = sorted(set(item[f"aliases_{language}"]))
    return {key: sorted(items.values(), key=lambda item: item["qid"]) for key, items in grouped.items()}


def collect(rows: list[dict], batch_size: int = 100, max_batches: int | None = None) -> dict:
    keys = sorted({row["InChIKey"] for row in rows})
    batches = [keys[start:start + batch_size] for start in range(0, len(keys), batch_size)]
    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    timestamps = {}
    failures = []
    requests_made = 0
    cached_batches = 0
    for index, batch in enumerate(batches):
        if max_batches is not None and index >= max_batches:
            break
        query = make_query(batch)
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cache_path = cache_dir / f"{query_hash}.json"
        try:
            if cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                if payload["query_sha256"] != query_hash or payload["keys"] != batch:
                    raise ValueError("Invalid coverage cache.")
                cached_batches += 1
            else:
                headers = {**_request_headers(wikimedia=True), "Accept": "application/sparql-results+json"}
                started = time.monotonic()
                response = _get(ENDPOINT, wikimedia=True, params={"query": query, "format": "json"}, headers=headers, timeout=(10, 65))
                requests_made += 1
                response.raise_for_status()
                body = response.json()
                bindings = body["results"]["bindings"]
                merge_bindings(batch, bindings)  # Validate before caching success.
                payload = {"query_sha256": query_hash, "keys": batch, "bindings": bindings,
                           "retrieved_on": datetime.now(timezone.utc).isoformat(), "endpoint": ENDPOINT,
                           "response_sha256": hashlib.sha256(response.content).hexdigest()}
                cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                # WDQS grants at most 60 seconds of processing in a minute.
                # A sequential client waits at least its request duration.
                time.sleep(max(2.0, time.monotonic() - started))
            results.update(merge_bindings(batch, payload["bindings"]))
            timestamps.update({key: payload["retrieved_on"] for key in batch})
            print(f"Batch {index + 1}/{len(batches)}: checked {len(results)}/{len(keys)} unique keys", flush=True)
        except Exception as exc:
            # Do not log response bodies, headers, contact information or keys.
            failures.append({"batch": index + 1, "key_count": len(batch), "error_type": type(exc).__name__})
            print(f"Batch {index + 1}: failed ({type(exc).__name__}); incomplete audit", flush=True)
            break

    links = []
    for row in rows:
        key = row["InChIKey"]
        entities = results.get(key, [])
        status = "not_checked" if key not in results else "not_found" if not entities else "matched_unique" if len(entities) == 1 else "ambiguous"
        links.append({"compound_id": row["ID"], "inchikey": key, "match_status": status,
                      "retrieved_on": timestamps.get(key, ""), "license": "CC0-1.0", "entities": entities})
    # A partial audit cannot replace a previously completed live alias map.
    complete = len(results) == len(keys)
    output_path = LINKS_PATH if complete else LINKS_PATH.with_name("wikidata_compound_links.partial.jsonl")
    temporary_path = output_path.with_suffix(".jsonl.tmp")
    temporary_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in links), encoding="utf-8")
    temporary_path.replace(output_path)
    counts = Counter(item["match_status"] for item in links)
    unique = [item for item in links if item["match_status"] == "matched_unique"]
    summary = {"train_rows": len(rows), "unique_inchikeys": len(keys), "checked_unique_inchikeys": len(results),
               "complete": complete, "match_status_counts": dict(counts),
               "unique_match_with_en_wikipedia": sum(bool(item["entities"][0]["en_wikipedia"]) for item in unique),
               "unique_match_with_ko_wikipedia": sum(bool(item["entities"][0]["ko_wikipedia"]) for item in unique),
               "unique_match_with_any_wikipedia": sum(bool(item["entities"][0]["en_wikipedia"] or item["entities"][0]["ko_wikipedia"]) for item in unique),
               "unique_match_with_korean_label": sum(bool(item["entities"][0]["label_ko"]) for item in unique),
               "network_batches": requests_made, "cached_batches": cached_batches, "failures": failures,
               "output": output_path.relative_to(compound_data.PROJECT_DIR).as_posix(), "gms_calls": 0,
               "document_text_collected": False, "document_approval": False,
               "endpoint": ENDPOINT, "license_source": "https://www.wikidata.org/wiki/Wikidata:Licensing"}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_PATH if complete else REPORT_PATH.with_name("wikidata_coverage_summary.partial.json")
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["compound_id", "inchikey", "match_status", "qid", "label_en", "label_ko", "en_wikipedia", "ko_wikipedia"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in links:
            entity = item["entities"][0] if item["match_status"] == "matched_unique" else {}
            writer.writerow({**{field: item[field] for field in fields[:3]}, **{field: entity.get(field, "") for field in fields[3:]}})
    if complete:
        write_document_candidates(rows, unique)
    return summary


def write_document_candidates(rows: list[dict], unique: list[dict]) -> None:
    """Choose 100 by document availability, never by source-group balance."""
    row_map = {row["ID"]: row for row in rows}
    candidates = [item for item in unique if item["entities"][0]["en_wikipedia"] or item["entities"][0]["ko_wikipedia"]]
    candidates.sort(key=lambda item: (
        not bool(item["entities"][0]["ko_wikipedia"]),
        not bool(item["entities"][0]["label_ko"]),
        item["entities"][0]["label_en"].casefold(), item["compound_id"],
    ))
    with (compound_data.DATA_DIR / "wikipedia_document_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["AqSolDB_ID", "Name", "InChIKey", "Wikidata_ID", "label_ko", "label_en", "document_url", "language", "identity_match", "document_status", "revision_id", "attribution_url"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates[:100]:
            entity = item["entities"][0]
            language = "ko" if entity["ko_wikipedia"] else "en"
            writer.writerow({
                "AqSolDB_ID": item["compound_id"], "Name": row_map[item["compound_id"]]["Name"],
                "InChIKey": item["inchikey"], "Wikidata_ID": entity["qid"],
                "label_ko": entity["label_ko"], "label_en": entity["label_en"],
                "document_url": entity[f"{language}_wikipedia"], "language": language,
                "identity_match": "exact_unique_InChIKey", "document_status": "link_only_pending_fetch",
                "revision_id": "", "attribution_url": "",
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=100, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--max-batches", type=int, help="Smoke check; incomplete results never become the live alias map.")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    _request_headers(wikimedia=True)  # Require real configured operator contact before any network call.
    rows = [row for row in compound_data.load_compounds() if row["_split"] == "train"]
    summary = collect(rows, args.batch_size, args.max_batches)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
