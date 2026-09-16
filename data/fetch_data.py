#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fetch_data.py  --  Paper 16 external-validity data acquisition (REAL data only).

Downloads:
  1. WHO FluNet virological surveillance (VIW_FNT)           -> data/flunet_raw.csv
  2. WHO FluNet metadata (VIW_FLU_METADATA)                  -> data/flunet_metadata.csv
  3. CDC FluView ILI surveillance via CMU DELPHI Epidata API -> data/fluview_raw.csv

NO synthetic / hand-constructed data is ever written. If a request fails the
script raises and reports the HTTP status; nothing is faked.

Re-runnable:  python fetch_data.py
Access date recorded in DATA_NOTES.md (2026-09-15).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ACCESS_DATE = "2026-09-15"

# ---------------------------------------------------------------------------
# URLs / parameters (all public, no registration / no key)
# ---------------------------------------------------------------------------
FLUNET_FNT_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
FLUNET_META_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FLU_METADATA?$format=csv"

DELPHI_FLUVIEW_URL = "https://api.delphi.cmu.edu/epidata/fluview/"

# Regions: national + all 10 HHS regions + 10 large states.
FLUVIEW_REGIONS = [
    "nat",
    "hhs1", "hhs2", "hhs3", "hhs4", "hhs5",
    "hhs6", "hhs7", "hhs8", "hhs9", "hhs10",
    "ca", "tx", "ny", "fl", "il", "pa", "oh", "ga", "nc", "mi",
]
# Epiweek range: MMWR week 30 of 2015 .. week 30 of 2025  (covers seasons 2015-16 .. 2024-25)
FLUVIEW_EPIWEEKS = "201530-202530"

TIMEOUT = 120
MAX_RETRIES = 5

# probe log for DATA_NOTES.md
PROBE_LOG: list[dict] = []


def _log(url: str, status, nbytes=None, note=""):
    PROBE_LOG.append(
        {"url": url, "http_status": status, "bytes": nbytes, "note": note}
    )


def get_binary(url: str, *, params=None, note="") -> bytes:
    """GET with retries + timeout. Returns raw bytes; raises on failure."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT,
                             headers={"User-Agent": "paper16-research/1.0"})
            if r.status_code == 200:
                _log(r.url, 200, len(r.content), note)
                return r.content
            last = f"HTTP {r.status_code}"
            _log(r.url, r.status_code, len(r.content), note + " (non-200)")
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            _log(url, "ERR", None, note + f" ({last})")
        time.sleep(2 * attempt)  # linear backoff
    raise RuntimeError(f"GET failed after {MAX_RETRIES} retries: {url} :: {last}")


def fetch_flunet():
    print("[1/3] WHO FluNet VIW_FNT ...")
    data = get_binary(FLUNET_FNT_URL, note="FluNet virological surveillance (full file)")
    out = HERE / "flunet_raw.csv"
    out.write_bytes(data)
    print(f"      -> {out} ({len(data):,} bytes)")

    print("[2/3] WHO FluNet metadata ...")
    meta = get_binary(FLUNET_META_URL, note="FluNet data dictionary")
    (HERE / "flunet_metadata.csv").write_bytes(meta)
    print(f"      -> flunet_metadata.csv ({len(meta):,} bytes)")


def fetch_fluview():
    print("[3/3] CDC FluView via DELPHI Epidata ...")
    rows: list[dict] = []
    # one call for all regions (API accepts comma-separated list)
    content = get_binary(
        DELPHI_FLUVIEW_URL,
        params={"regions": ",".join(FLUVIEW_REGIONS), "epiweeks": FLUVIEW_EPIWEEKS},
        note=f"FluView regions={len(FLUVIEW_REGIONS)} epiweeks={FLUVIEW_EPIWEEKS}",
    )
    payload = json.loads(content)
    if payload.get("result") != 1:
        raise RuntimeError(f"DELPHI result != 1: {payload.get('message')}")
    rows.extend(payload["epidata"])
    print(f"      batches returned {len(rows):,} raw rows")

    import pandas as pd
    df = pd.DataFrame(rows)
    out = HERE / "fluview_raw.csv"
    df.to_csv(out, index=False)
    print(f"      -> {out} ({len(df):,} rows, {out.stat().st_size:,} bytes)")
    return df


def main():
    print(f"Access date: {ACCESS_DATE}")
    fetch_flunet()
    fetch_fluview()
    (HERE / "fetch_probe_log.json").write_text(
        json.dumps({"access_date": ACCESS_DATE, "probes": PROBE_LOG}, indent=2),
        encoding="utf-8",
    )
    print("Done. Probe log -> data/fetch_probe_log.json")


if __name__ == "__main__":
    sys.exit(main())
