#!/usr/bin/env python3
"""
STRING v12.0 Re-validation of Dark Matter Pairs (Step 53)
=========================================================

Query all 44 dark matter pairs against STRING v12.0 API to check if
any have gained evidence since v11.5.

Output: results/string_v12_revalidation.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_results_dir, SEED

RESULTS = get_results_dir()

BANNER = "=" * 64
API_BASE = "https://string-db.org/api/tsv"
TAXON = 4932
DELAY = 1.2  # seconds between requests (respect rate limits)


def load_dark_matter_pairs():
    """Load the 44 dark matter pairs from existing results."""
    path = RESULTS / "functional_dark_matter.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["top_100_catalog"]


def query_string_pair(id_a, id_b):
    """Query STRING v12.0 API for a single protein pair."""
    params = urllib.parse.urlencode({
        "identifiers": f"{id_a}\r{id_b}",
        "species": TAXON,
        "required_score": 0,
        "network_type": "functional",
    })
    url = f"{API_BASE}/network?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "GF-consistency-framework/2.3.1")
        with urllib.request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8")

        lines = text.strip().split("\n")
        if len(lines) <= 1:
            # Only header, no interaction data
            return None

        # Parse first data line
        header = lines[0].split("\t")
        row = dict(zip(header, lines[1].split("\t")))
        return {
            "combined_score": int(float(row.get("score", 0))),
            "nscore": int(float(row.get("nscore", 0))),
            "fscore": int(float(row.get("fscore", 0))),
            "pscore": int(float(row.get("pscore", 0))),
            "ascore": int(float(row.get("ascore", 0))),
            "escore": int(float(row.get("escore", 0))),
            "dscore": int(float(row.get("dscore", 0))),
            "tscore": int(float(row.get("tscore", 0))),
            "preferred_a": row.get("preferredName_A", ""),
            "preferred_b": row.get("preferredName_B", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def resolve_string_id(orf_name):
    """Resolve ORF name to STRING ID via API."""
    params = urllib.parse.urlencode({
        "identifiers": orf_name,
        "species": TAXON,
    })
    url = f"{API_BASE}/get_string_ids?{params}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "GF-consistency-framework/2.3.1")
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode("utf-8")

        lines = text.strip().split("\n")
        if len(lines) >= 2:
            header = lines[0].split("\t")
            row = dict(zip(header, lines[1].split("\t")))
            return row.get("stringId", ""), row.get("preferredName", "")
        return f"{TAXON}.{orf_name}", orf_name
    except Exception as e:
        return f"{TAXON}.{orf_name}", orf_name


def run():
    t_start = time.time()
    print(BANNER)
    print("  Phase 20: STRING v12.0 Re-validation of Dark Matter Pairs")
    print(BANNER)

    pairs = load_dark_matter_pairs()
    print(f"  Loaded {len(pairs)} dark matter pairs")

    # Cache for ID resolution
    id_cache = {}
    name_cache = {}

    results = []
    gained_evidence = 0
    gained_high_conf = 0
    errors = 0

    for i, pair in enumerate(pairs):
        pa = pair["protein_a"]
        pb = pair["protein_b"]
        v11_score = pair.get("string_score", 0)

        # Resolve IDs
        if pa not in id_cache:
            string_id, pref_name = resolve_string_id(pa)
            id_cache[pa] = string_id
            name_cache[pa] = pref_name
            time.sleep(0.3)

        if pb not in id_cache:
            string_id, pref_name = resolve_string_id(pb)
            id_cache[pb] = string_id
            name_cache[pb] = pref_name
            time.sleep(0.3)

        # Query pair
        v12_result = query_string_pair(id_cache[pa], id_cache[pb])
        time.sleep(DELAY)

        v12_score = 0
        v12_channels = {}
        status = "no_evidence"

        if v12_result is None:
            v12_score = 0
            status = "no_evidence"
        elif "error" in v12_result:
            status = f"error: {v12_result['error']}"
            errors += 1
        else:
            v12_score = v12_result["combined_score"]
            v12_channels = {
                "neighborhood": v12_result["nscore"],
                "fusion": v12_result["fscore"],
                "cooccurrence": v12_result["pscore"],
                "coexpression": v12_result["ascore"],
                "experiments": v12_result["escore"],
                "database": v12_result["dscore"],
                "textmining": v12_result["tscore"],
            }
            if v12_score >= 700:
                status = "gained_high_confidence"
                gained_high_conf += 1
                gained_evidence += 1
            elif v12_score >= 400:
                status = "gained_low_confidence"
                gained_evidence += 1
            elif v12_score > v11_score:
                status = "gained_weak_evidence"
                gained_evidence += 1
            elif v12_score == v11_score:
                status = "unchanged"
            else:
                status = "score_decreased"

        entry = {
            "protein_a": pa,
            "protein_b": pb,
            "preferred_a": name_cache.get(pa, ""),
            "preferred_b": name_cache.get(pb, ""),
            "string_id_a": id_cache.get(pa, ""),
            "string_id_b": id_cache.get(pb, ""),
            "v11_combined_score": v11_score,
            "v12_combined_score": v12_score,
            "v12_channels": v12_channels,
            "status": status,
            "shared_go_terms": pair.get("shared_go_terms", []),
            "confidence_score": pair.get("confidence_score", 0),
        }
        results.append(entry)

        completed = i + 1
        elapsed = time.time() - t_start
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(pairs) - completed) / rate if rate > 0 else 0
        status_mark = "!" if "gained" in status else "."
        print(f"  [{completed:2d}/{len(pairs)}] {pa}--{pb}: "
              f"v11={v11_score:3d} -> v12={v12_score:3d} "
              f"[{status}] {status_mark}  "
              f"({rate:.1f}/s, ETA {eta:.0f}s)")

    # ---- Summary ----
    elapsed = time.time() - t_start
    unchanged = sum(1 for r in results if r["status"] == "unchanged")
    no_evidence = sum(1 for r in results if r["status"] == "no_evidence")

    output = {
        "description": "STRING v12.0 Re-validation of Dark Matter Pairs",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_base": API_BASE,
        "taxon": TAXON,
        "n_pairs": len(pairs),
        "summary": {
            "no_evidence_v12": no_evidence,
            "unchanged": unchanged,
            "gained_any_evidence": gained_evidence,
            "gained_high_confidence": gained_high_conf,
            "errors": errors,
        },
        "conclusion": (
            f"Of {len(pairs)} dark matter pairs, {no_evidence} remain completely "
            f"invisible in STRING v12.0, {unchanged} have unchanged scores, "
            f"{gained_evidence} gained some evidence, and "
            f"{gained_high_conf} reached high-confidence (>= 700). "
            f"{'The dark matter catalog is robust to database updates.' if gained_high_conf == 0 else 'WARNING: some pairs gained high-confidence status.'}"
        ),
        "pairs": results,
    }

    out_file = RESULTS / "string_v12_revalidation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  RESULTS:")
    print(f"  No evidence in v12.0:    {no_evidence}")
    print(f"  Unchanged:               {unchanged}")
    print(f"  Gained any evidence:     {gained_evidence}")
    print(f"  Gained high-confidence:  {gained_high_conf}")
    print(f"  Errors:                  {errors}")
    print(f"  Conclusion: {output['conclusion']}")
    print(f"\n  Saved to {out_file}")
    print(f"  Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    run()
