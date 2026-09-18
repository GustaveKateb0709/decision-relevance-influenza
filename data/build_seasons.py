#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_seasons.py  --  Build the season panel data/seasons.csv from REAL raw data.

Inputs :  data/flunet_raw.csv   (WHO FluNet VIW_FNT)
          data/fluview_raw.csv  (CDC FluView via CMU DELPHI Epidata)

Season definition (Northern Hemisphere):
    week 40 of year Y  ->  week 20 of year Y+1,  labelled "{Y}-{Y+1}"
    week_index = 1 .. ~33, counted in *actual 7-day steps* from the season's
    week-40 start date (so 52- vs 53-week years stay correct).
    Off-season weeks (21..39) are dropped.

Screening rule (per source x region x season unit):
    keep only if  (a) >= 8 weeks present in the season window, AND
                  (b) the primary outcome is observed in >= 8 of those weeks, AND
                  (c) the primary outcome is NOT identically zero.
    Primary outcome: virus_positives (FluNet) / ili_count (FluView).

NO synthetic data. All numbers keep original precision.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ACCESS_DATE = "2026-09-15"

SEASON_START_WEEK = 40
SEASON_END_WEEK = 20
SEASON_START_YEARS = list(range(2015, 2025))  # -> seasons 2015-16 .. 2024-25 (10 seasons)

# reproducible FluNet quality threshold
FLUNET_MIN_WEEKS = 200
FLUNET_MIN_INFALL_OBS = 200
FLUNET_MIN_SPECIMENS = 20_000

FLUVIEW_REGION_NAMES = {
    "nat": "United States (national)",
    "hhs1": "HHS Region 1 (Boston)", "hhs2": "HHS Region 2 (New York)",
    "hhs3": "HHS Region 3 (Philadelphia)", "hhs4": "HHS Region 4 (Atlanta)",
    "hhs5": "HHS Region 5 (Chicago)", "hhs6": "HHS Region 6 (Dallas)",
    "hhs7": "HHS Region 7 (Kansas City)", "hhs8": "HHS Region 8 (Denver)",
    "hhs9": "HHS Region 9 (San Francisco)", "hhs10": "HHS Region 10 (Seattle)",
    "ca": "California", "tx": "Texas", "ny": "New York (state)", "fl": "Florida",
    "il": "Illinois", "pa": "Pennsylvania", "oh": "Ohio", "ga": "Georgia",
    "nc": "North Carolina", "mi": "Michigan",
}


# ---------------------------------------------------------------------------
# calendar helpers
# ---------------------------------------------------------------------------
def iso_week_monday(year: int, week: int) -> dt.date:
    """Monday of ISO week `week` of `year`."""
    return dt.date.fromisocalendar(year, week, 1)


def mmwr_week1_start(year: int) -> dt.date:
    """Sunday on which MMWR/CDC week 1 of `year` begins.

    MMWR week 1 = first Sun-Sat week containing >=4 days of the new year.
    """
    jan1 = dt.date(year, 1, 1)
    days_since_sun = (jan1.weekday() + 1) % 7          # Mon=0..Sun=6 -> days since last Sunday
    prev_sunday = jan1 - dt.timedelta(days=days_since_sun)
    days_of_year_in_that_week = 7 - days_since_sun
    if days_of_year_in_that_week >= 4:
        return prev_sunday
    return prev_sunday + dt.timedelta(days=7)


def mmwr_week_start(epiweek: int) -> dt.date:
    """Sunday-start date for an MMWR epiweek int YYYYWW."""
    y, w = divmod(epiweek, 100)
    return mmwr_week1_start(y) + dt.timedelta(days=(w - 1) * 7)


def season_label_from_start_year(y: int) -> str:
    return f"{y}-{y + 1}"


# ---------------------------------------------------------------------------
# FluNet
# ---------------------------------------------------------------------------
def load_flunet() -> pd.DataFrame:
    fn = pd.read_csv(HERE / "flunet_raw.csv", low_memory=False)
    num = ["SPEC_PROCESSED_NB", "SPEC_RECEIVED_NB", "INF_A", "INF_B", "INF_ALL",
           "INF_NEGATIVE", "RSV"]
    fn = fn[fn.HEMISPHERE == "NH"].copy()

    # quality screen on the 2015-2025 season windows (independent of final seasons)
    win = fn[fn.ISO_YEAR.between(2015, 2025) & ((fn.ISO_WEEK >= 40) | (fn.ISO_WEEK <= 20))]
    agg0 = (win.groupby(["COUNTRY_CODE", "ISO_YEAR", "ISO_WEEK"], as_index=False)[num]
               .sum(min_count=1))
    cov = agg0.groupby("COUNTRY_CODE").agg(
        n_weeks=("ISO_WEEK", "size"),
        infall_obs=("INF_ALL", "count"),
        spec_sum=("SPEC_PROCESSED_NB", "sum"),
    ).reset_index()
    cov = cov[~cov.COUNTRY_CODE.str.match(r"^X")]
    keep = cov[(cov.n_weeks >= FLUNET_MIN_WEEKS) &
               (cov.infall_obs >= FLUNET_MIN_INFALL_OBS) &
               (cov.spec_sum >= FLUNET_MIN_SPECIMENS)].COUNTRY_CODE
    fn = fn[fn.COUNTRY_CODE.isin(set(keep))].copy()

    # collapse ORIGIN_SOURCE (SENTINEL / NONSENTINEL / NOTDEFINED) to country-week totals
    agg = (fn.groupby(["COUNTRY_CODE", "COUNTRY_AREA_TERRITORY", "ISO_YEAR", "ISO_WEEK",
                       "ISO_WEEKSTARTDATE"], as_index=False)[num].sum(min_count=1))
    agg["week_start_date"] = pd.to_datetime(agg.ISO_WEEKSTARTDATE)

    # season + week_index
    agg["season_start_year"] = np.where(agg.ISO_WEEK >= SEASON_START_WEEK,
                                        agg.ISO_YEAR, agg.ISO_YEAR - 1)
    agg = agg[agg.season_start_year.isin(SEASON_START_YEARS)]
    agg = agg[(agg.ISO_WEEK >= SEASON_START_WEEK) | (agg.ISO_WEEK <= SEASON_END_WEEK)]
    agg["season"] = agg.season_start_year.map(season_label_from_start_year)
    w40 = {y: iso_week_monday(y, SEASON_START_WEEK) for y in SEASON_START_YEARS}
    agg["week_index"] = [
        (d.date() - w40[sy]).days // 7 + 1
        for d, sy in zip(agg.week_start_date, agg.season_start_year)
    ]

    out = pd.DataFrame({
        "source": "flunet",
        "region": agg.COUNTRY_CODE,
        "region_name": agg.COUNTRY_AREA_TERRITORY,
        "season": agg.season,
        "week_index": agg.week_index,
        "week_start_date": agg.week_start_date.dt.date,
        "epiweek": agg.ISO_YEAR * 100 + agg.ISO_WEEK,
        "ili_count": np.nan,
        "total_patients": np.nan,
        "wili": np.nan,
        "ili": np.nan,
        "virus_positives": agg.INF_ALL,
        "virus_positives_a": agg.INF_A,
        "virus_positives_b": agg.INF_B,
        "specimens_processed": agg.SPEC_PROCESSED_NB,
        "num_providers": np.nan,
    })
    out["positivity"] = out.virus_positives / out.specimens_processed.replace(0, np.nan)
    # FluNet artifact: non-sentinel labs sometimes report positives with 0/missing
    # processed specimens, giving positivity>1 (biologically impossible). Flag, never alter.
    out["positivity_invalid"] = (out.positivity > 1) | (out.positivity < 0)
    return out


# ---------------------------------------------------------------------------
# FluView
# ---------------------------------------------------------------------------
def load_fluview() -> pd.DataFrame:
    fv = pd.read_csv(HERE / "fluview_raw.csv")
    fv["week_start_date"] = pd.to_datetime(fv.epiweek.map(mmwr_week_start))
    fv["mmwr_year"] = fv.epiweek // 100
    fv["mmwr_week"] = fv.epiweek % 100
    fv["season_start_year"] = np.where(fv.mmwr_week >= SEASON_START_WEEK,
                                       fv.mmwr_year, fv.mmwr_year - 1)
    fv = fv[fv.season_start_year.isin(SEASON_START_YEARS)]
    fv = fv[(fv.mmwr_week >= SEASON_START_WEEK) | (fv.mmwr_week <= SEASON_END_WEEK)]
    fv["season"] = fv.season_start_year.map(season_label_from_start_year)
    w40 = {y: mmwr_week_start(y * 100 + SEASON_START_WEEK) for y in SEASON_START_YEARS}
    fv["week_index"] = [
        (d.date() - w40[sy]).days // 7 + 1
        for d, sy in zip(fv.week_start_date, fv.season_start_year)
    ]

    out = pd.DataFrame({
        "source": "fluview",
        "region": fv.region,
        "region_name": fv.region.map(FLUVIEW_REGION_NAMES).fillna(fv.region),
        "season": fv.season,
        "week_index": fv.week_index,
        "week_start_date": fv.week_start_date.dt.date,
        "epiweek": fv.epiweek,
        "ili_count": fv.num_ili,
        "total_patients": fv.num_patients,
        "wili": fv.wili,
        "ili": fv.ili,
        "virus_positives": np.nan,
        "virus_positives_a": np.nan,
        "virus_positives_b": np.nan,
        "specimens_processed": np.nan,
        "num_providers": fv.num_providers,
        "positivity": np.nan,
        "positivity_invalid": False,
    })
    return out


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------
def screen(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    recs, dropped = [], []
    for (src, reg, sea), g in df.groupby(["source", "region", "season"]):
        primary = "virus_positives" if src == "flunet" else "ili_count"
        n_weeks = len(g)
        n_obs = g[primary].notna().sum()
        total = g[primary].sum(skipna=True)
        ok = (n_weeks >= 8) and (n_obs >= 8) and (total > 0)
        rec = {"source": src, "region": reg, "season": sea,
               "n_weeks": n_weeks, "n_obs": int(n_obs), "sum_primary": float(total),
               "kept": ok}
        recs.append(rec)
        if not ok:
            dropped.append(rec)
    audit = pd.DataFrame(recs).sort_values(["source", "region", "season"]).reset_index(drop=True)
    keep_mask = audit.set_index(["source", "region", "season"]).kept
    idx = pd.MultiIndex.from_frame(df[["source", "region", "season"]])
    df = df[keep_mask.reindex(idx).values].copy()
    return df, audit, pd.DataFrame(dropped)


def main():
    fl = load_flunet()
    fv = load_fluview()
    combined = pd.concat([fl, fv], ignore_index=True)

    combined, audit, dropped = screen(combined)
    combined = combined.sort_values(["source", "region", "season", "week_index"]).reset_index(drop=True)

    cols = ["source", "region", "region_name", "season", "week_index", "epiweek",
            "week_start_date", "ili_count", "total_patients", "wili", "ili",
            "virus_positives", "virus_positives_a", "virus_positives_b",
            "specimens_processed", "num_providers", "positivity", "positivity_invalid"]
    combined = combined[cols]
    combined.to_csv(HERE / "seasons.csv", index=False)
    audit.to_csv(HERE / "seasons_audit.csv", index=False)
    bal = (combined.groupby(["source", "region", "season"]).week_index
           .agg(n_weeks="size")
           .assign(contiguous=lambda d: d.n_weeks.between(32, 34))
           .reset_index())
    bal.to_csv(HERE / "seasons_balance.csv", index=False)

    print(f"seasons.csv rows: {len(combined):,}  (dropped units: {len(dropped)})")
    print(combined.groupby("source").agg(rows=("week_index", "size"),
                                         units=("season", lambda s: s.size)))
    summ = (combined.groupby(["source", "region", "season"]).size()
            .reset_index(name="n_weeks"))
    print("\nkept units:", len(summ))
    print(summ.groupby("source").size())
    json.dump({
        "access_date": ACCESS_DATE,
        "n_rows": int(len(combined)),
        "n_units": int(len(summ)),
        "flunet_countries": sorted(fl.region.unique().tolist()),
        "fluview_regions": sorted(fv.region.unique().tolist()),
    }, open(HERE / "seasons_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()
