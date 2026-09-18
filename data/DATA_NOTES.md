# DATA_NOTES.md — external-validity surveillance data

**Access date (all downloads): 2026-09-15**
**Working dir:** `data/`
**Scripts:** `fetch_data.py` (download, re-runnable) → `build_seasons.py` (panel build, re-runnable)

> **Honesty declaration.** Every file in this directory is a **real** fetch/parse of the
> public endpoints listed below. **No value was hand-constructed, imputed, or synthesised.**
> All three downloads succeeded on the first attempt (HTTP 200). Nothing was substituted or
> mirrored because nothing failed. Raw numeric precision is preserved (no rounding).

---

## 1. Access log (URL · HTTP status · size · result)

| # | Endpoint | Params | HTTP | Bytes | Output file | Rows |
|---|----------|--------|------|-------|-------------|------|
| 1 | `https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv` | — (full file) | **200** | 32,038,437 | `flunet_raw.csv` | 188,389 data rows (196,176 physical lines) |
| 2 | `https://xmart-api-public.who.int/FLUMART/VIW_FLU_METADATA?$format=csv` | — | **200** | 15,241 | `flunet_metadata.csv` | 90 field-definition rows |
| 3 | `https://api.delphi.cmu.edu/epidata/fluview/` | `regions=nat,hhs1..hhs10,ca,tx,ny,fl,il,pa,oh,ga,nc,mi` ; `epiweeks=201530-202530` | **200** | 890,219 | `fluview_raw.csv` | 10,639 rows |

- Machine-readable probe log: `fetch_probe_log.json` (`access_date = 2026-09-15`).
- No registration, no API key was required for any endpoint.
- `flunet_raw.csv` reports 196,176 *physical* lines but 188,389 *logical* rows: some text
  fields (`WCR_COMMENT`, `LAB_RESULT_COMMENT`, …) contain embedded newlines. Parse with a CSV
  reader (`pd.read_csv`), never by line count.

---

## 2. Field dictionary

### 2.1 FluNet (`flunet_raw.csv`) — WHO virological surveillance
Full data dictionary scraped to `flunet_metadata.csv` (columns `DatasetName, TableName,
FieldName, DataType, Description, Comments`). Fields used downstream:

| Field | Meaning | Unit / notes |
|-------|---------|--------------|
| `COUNTRY_CODE` | ISO3 country code (WHO-defined codes where no ISO3) | — |
| `COUNTRY_AREA_TERRITORY` | country / area name | — |
| `HEMISPHERE` | `NH` / `SH` | we keep **NH only** |
| `ISO_YEAR`, `ISO_WEEK`, `ISO_WEEKSTARTDATE` | ISO-8601 week of the Monday week-start | Monday-based week |
| `MMWR_YEAR`, `MMWR_WEEK`, `MMWR_WEEKSTARTDATE` | US MMWR week (Sunday-based) | provided by WHO alongside ISO |
| `ORIGIN_SOURCE` | `SENTINEL` / `NONSENTINEL` / `NOTDEFINED` | split reporting streams |
| `SPEC_PROCESSED_NB` | specimens processed for influenza | count |
| `INF_A`, `INF_B` | influenza A / B positives | count |
| `INF_ALL` | all influenza positives (`= A + B` in practice) | count |
| `INF_NEGATIVE` | specimens testing negative | count |
| `RSV` | RSV positives | count |
| `ILI_ACTIVITY` | qualitative ILI activity code | **categorical, mostly missing — not used** |

**FluNet has NO ILI case counts and NO total-patient denominator.** It is a *virological*
stream only (positives + specimens processed). `ILI_ACTIVITY` is a text code, not a count.

### 2.2 FluView (`fluview_raw.csv`) — CDC outpatient ILI surveillance

| Field | Meaning | Unit |
|-------|---------|------|
| `region` | `nat`, `hhs1..hhs10`, or state postal code | — |
| `epiweek` | MMWR week `YYYYWW` | Sunday-based |
| `issue` | MMWR week in which the record was published | data vintage |
| `lag` | `issue − epiweek` | weeks (revision lag) |
| `num_ili` | ILI visits (outpatient) | count |
| `num_patients` | total outpatient visits (denominator) | count |
| `num_providers` | number of reporting providers | count |
| `wili` | **weighted** ILI % = ILI / total visits | percent |
| `ili` | unweighted ILI % | percent |
| `num_age_0..5` | ILI by age band | counts; `num_age_2` (5–17y) is often `null` |

`ili_count = num_ili`, `total_patients = num_patients` in the panel.

### 2.3 Panel (`seasons.csv`) — the file the paper should use

One row = one `source × region × season × week`. Columns:
`source, region, region_name, season, week_index, epiweek, week_start_date, ili_count,
total_patients, wili, ili, virus_positives, virus_positives_a, virus_positives_b,
specimens_processed, num_providers, positivity, positivity_invalid`.
Columns not applicable to a source are `NaN` (e.g. `ili_count` for FluNet; `virus_positives`
for FluView). `positivity = virus_positives / specimens_processed` (FluNet only).

---

## 3. Season definition, screening, and final panel

### 3.1 Season definition (Northern Hemisphere)
**Week 40 of year Y → week 20 of year Y+1**, labelled `"Y-(Y+1)"`. Ten seasons are kept:
**2015-2016 … 2024-2025**.
- `week_index = 1 .. 33/34`, counted in real 7-day steps from the season's week-40 start date,
  so 52- vs 53-week calendar years stay correct (seasons starting 2015, 2020 have 34 weeks).
- Off-season weeks 21–39 are **excluded**.
- FluNet rows use the **ISO** week (`ISO_WEEK ≥ 40 or ≤ 20`); FluView rows use the **MMWR**
  week (`MMWR_WEEK ≥ 40 or ≤ 20`). The two week systems are *not* identical (see §5.1).

### 3.2 Screening rules (applied per `source × region × season` unit)
A unit is **kept only if all three hold**:
1. ≥ 8 weeks present inside the season window, **and**
2. the primary outcome observed in ≥ 8 of those weeks
   (primary outcome = `virus_positives` for FluNet, `ili_count` for FluView), **and**
3. the primary outcome is **not identically zero**.

### 3.3 FluNet region selection (reproducible threshold)
From all `HEMISPHERE == NH` countries, keep those with (in the 2015–2025 season windows):
`n_weeks ≥ 200` **and** weeks with `INF_ALL` observed `≥ 200` **and** total
`SPEC_PROCESSED_NB ≥ 20,000`; WHO aggregate codes matching `^X` are dropped.
This yields **64 countries**.

### 3.4 Final panel sizes (after screening)

| source | regions | units (region×season) | rows |
|--------|---------|----------------------|------|
| `fluview` | 21 | 204 | 6,752 |
| `flunet` | 64 | 591 | 19,084 |
| **total** | **85** | **795** | **25,836** |

47 units were dropped: **40 in the COVID-era season `2020-2021`** (influenza essentially
vanished globally: 0 or near-0 positives) and **7 in `2021-2022`** (`JPN, KHM, KOR, LAO, MLT,
SGP, THA` — each with < 8 observed weeks).
The `2020-2021` FluNet unit count collapses from ~64 to **23** countries. **This is signal,
not error** — it is a genuine feature of influenza epidemiology and should be handled
explicitly (e.g. as a structural break) in any season panel.

Per-unit week counts and completeness: **`seasons_balance.csv`** (authoritative listing).
Full keep/drop audit with reason columns: **`seasons_audit.csv`**.
Machine summary of region lists: **`seasons_meta.json`**.

### 3.5 Kept series list
- **FluView (21 regions × 10 seasons = 204 units, minus the 6 absent FL seasons):**
  `nat`, `hhs1`–`hhs10`, `ca, tx, ny, fl, il, pa, oh, ga, nc, mi`.
  All regions have 33–34 weeks per season **except `fl` (Florida)**, which only reports from
  MMWR week 202140 onward → FL units exist for **2021-2022 … 2024-2025 only** (132 weeks total).
- **FluNet (64 NH countries):** AFG, AUT, BGD, BHR, CAN, CHN, CIV, COL, CRI, CZE, DEU, DNK,
  EGY, ESP, EST, ETH, FRA, GHA, GRC, HKG, HRV, HUN, IND, IRL, IRN, ISL, ISR, ITA, JOR, JPN,
  KAZ, KHM, KOR, LAO, LKA, LTU, LUX, LVA, MDA, MEX, MLT, MNG, MYS, NIC, NLD, NOR, NPL, OMN,
  PAK, PAN, POL, PRT, QAT, ROU, RUS, SAU, SGP, SVK, SVN, SWE, THA, TUR, UKR, USA.
  - 20 countries contribute all 10 seasons; 39 contribute 9; 5 contribute 8
    (MLT, SGP, JPN, KHM, KOR — each loses both 2020-21 and 2021-22 to screening).

---

## 4. Key facts for modelling / external validity

- **Two independent surveillance streams** for overlapping geography: FluNet gives
  *virological* positivity (`INF_ALL/SPEC_PROCESSED_NB`) incl. USA; FluView gives *ILI
  consultation* rate (`wili`) for US national + 10 HHS regions + 10 states.
- **FluNet USA** aggregates three `ORIGIN_SOURCE` streams (`NOTDEFINED` pre-2016;
  `SENTINEL` + `NONSENTINEL` from ~2016). The builder **sums across `ORIGIN_SOURCE`** to a
  single country-week total (documented; the sentinel/non-sentinel split changes over time).
- Missingness is **structural, not random** (see §5.3). Do not naively complete-case.
- `wili` and `ili` are near-identical for many regions (weighting ~1 when provider coverage is
  homogeneous); both are provided.

## 5. Known limitations & pitfalls

1. **ISO vs MMWR week mismatch.** FluNet weeks start Monday, FluView/MMWR weeks start Sunday,
   and their "week 40" anchors can differ by up to ~6 days in a given year. The two sources are
   therefore **not week-for-week aligned**; align by *date* (`week_start_date`) or by matching
   `week_index`, not by raw week number. Both are, however, internally consistent across seasons.
2. **Data revision / vintage (FluView).** FluView is revised for weeks after release. The
   snapshot here carries `issue` and `lag` columns (lag ranged 51–114 weeks). This is the
   **latest vintage as of 2026-09-15**, not a real-time vintage; back-test/revision analysis
   would need the `issue=` parameter, which we did **not** pull.
3. **DELPHI Epidata "best-effort" maintenance.** The API self-describes non-COVID sources as
   best-effort; the FluView extractor can lag or change. Numbers are as-served on the access date.
4. **FluNet is voluntary / uneven.** Countries report at very different volumes and cadence;
   `ILI_ACTIVITY` is mostly missing; `SPEC_PROCESSED_NB` and `INF_ALL` are sometimes mutually
   inconsistent (§5.5). Coverage is right-skewed (`USA` ~23 M specimens vs ~20 k for small states).
5. **Biologically impossible positivity (flagged).** For **203 of 19,084** FluNet rows
   (`positivity > 1`; 165 of them `NLD`, i.e. year-round non-sentinel labs reporting positives
   with 0 processed specimens), `INF_ALL > SPEC_PROCESSED_NB`. We **did not alter** the raw
   counts; instead `positivity_invalid = TRUE` flags these rows. Analysts should drop or
   winsorise flagged rows before using `positivity`.
6. **Florida gap.** `fl` has no ILI data before MMWR 202140 (state reporting discontinuity).
7. **COVID seasons.** `2020-2021` (and to a lesser degree `2021-2022`) are anomalously
   low/absent. The panel therefore has a genuine **structural break**; do not treat seasons as
   exchangeable without addressing it.
8. **No ILI denominator in FluNet.** `total_patients` is `NaN` for all FluNet rows, and
   `virus_positives` is `NaN` for all FluView rows — the two streams measure different things.
9. **Age band `num_age_2` (5–17y)** is frequently `null` in FluView (verified in the payload);
   age-stratified work needs care.
10. **2025-2026 partial season excluded.** FluNet raw contains data into ISO 2026, but we froze
    the panel at **2024-2025** to keep 10 complete seasons aligned across both sources.

## 6. Reproduce

```bash
cd data
python3 fetch_data.py   # -> *_raw.csv, *_metadata.csv, fetch_probe_log.json
python3 build_seasons.py # -> seasons.csv, seasons_audit.csv, seasons_balance.csv, seasons_meta.json
```

## 7. File inventory

| File | What |
|------|------|
| `flunet_raw.csv` | raw WHO FluNet VIW_FNT, real download |
| `flunet_metadata.csv` | FluNet data dictionary |
| `fluview_raw.csv` | raw CDC FluView (DELPHI), real download |
| `seasons.csv` | **analysis panel** (25,836 rows) |
| `seasons_audit.csv` | per-unit keep/drop with reason counts |
| `seasons_balance.csv` | per-unit week counts / completeness |
| `seasons_meta.json` | region lists + row/unit counts |
| `fetch_probe_log.json` | URL · HTTP status · bytes log |
| `fetch_data.py`, `build_seasons.py` | re-runnable scripts |
| `../figures/fig_external_validity_seasons.png` | (a) US wili by season, (b) FluNet positivity 6 countries |
