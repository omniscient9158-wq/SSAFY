"""Resume the saved provenance review without refetching, approving, or indexing."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
PROVENANCE = ROOT / "research/provenance"
DECISIONS = EVIDENCE / "document_review_staging.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    snapshot = json.loads((PROVENANCE / "description_review_records.json").read_text(encoding="utf-8"))
    records = {item["compound_id"]: item for item in snapshot["records"]}
    pages = json.loads((PROVENANCE / "description_review/original_pages.json").read_text(encoding="utf-8"))
    triage = {row["AqSolDB_ID"]: row for row in read_csv(EVIDENCE / "document_candidate_triage.csv")}
    staged = {row["AqSolDB_ID"]: row for row in read_csv(DECISIONS)}
    dispositions = {row["AqSolDB_ID"]: row for row in read_csv(EVIDENCE / "document_review_dispositions_20261023.csv")}
    approved_ids = {row["AqSolDB_ID"] for row in read_csv(ROOT / "data/approved_external_sources.csv") if row["index_eligible"] == "yes"}
    manifest = read_csv(ROOT / "data/document_discovery_manifest.csv")
    rows = []
    for item in manifest:
        cid = item["AqSolDB_ID"]
        if cid not in records:
            continue
        source = records.get(cid, {})
        original_urls = sorted({a["source_url"] for a in source.get("annotations", [])
                                if a.get("provider") == "CAMEO Chemicals" and a.get("source_url")})
        available = [url for url in original_urls if pages.get(url, {}).get("http_status") == 200
                     and not pages[url].get("error")]
        decision = staged.get(cid)
        if decision:
            # Staging is evidence of review, never an implicit index approval.
            assert decision["index_eligible"] == "no"
            assert source.get("identity_matches") is True
            assert source.get("property_inchikey") == item["InChIKey"]
            assert decision["source_url"] in available
            text = pages[decision["source_url"]]["text"]
            assert hashlib.sha256(text.encode("utf-8")).hexdigest() == decision["source_text_sha256"]
            assert decision["raw_excerpt"].split("\n") == text[text.index("Water Solubility:"):].splitlines()[:4]
        rows.append({
            "AqSolDB_ID": cid,
            "name": item["Name"],
            "source_record_cached": str(bool(source) and not source.get("error")),
            "pubchem_identity_matches": str(source.get("identity_matches", False)),
            "cameo_original_cached": str(bool(available)),
            "previous_triage_status": triage.get(cid, {}).get("triage_status", ""),
            "excerpt_review_stage": dispositions[cid]["disposition"],
            "reviewed_excerpt_source": decision["source_url"] if decision else dispositions[cid]["cached_original_urls"],
            "index_eligible": "yes" if cid in approved_ids else "no",
        })
    target = EVIDENCE / "document_review_progress.csv"
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_snapshot_assembled_on": snapshot.get("assembled_on"),
        "reviewed_candidate_scope": len(rows),
        "pending_approval_compounds": sum(r["index_eligible"] != "yes" for r in rows),
        "approved_compounds_total": len(approved_ids),
        "source_records_cached": sum(r["source_record_cached"] == "True" for r in rows),
        "pubchem_identity_matches": sum(r["pubchem_identity_matches"] == "True" for r in rows),
        "original_pages_cached": sum(p.get("http_status") == 200 and not p.get("error") for p in pages.values()),
        "original_pages_failed": sum(bool(p.get("error")) for p in pages.values()),
        "compounds_with_cached_cameo_original": sum(r["cameo_original_cached"] == "True" for r in rows),
        "staged_excerpt_reviews": len(staged),
        "review_stage_counts": dict(Counter(r["excerpt_review_stage"] for r in rows)),
        "new_approved_documents": sum(r["index_eligible"] == "yes" for r in rows),
        "network_calls": 0,
        "model_calls": 0,
        "scope": "Preserved 94-candidate snapshot review. Approval follows the bounded-excerpt decision CSV; approval and vector indexing are separate stages.",
    }
    (EVIDENCE / "document_review_progress.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
