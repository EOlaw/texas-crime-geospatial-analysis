# Data Sources

## 1. Texas DPS UCR (Socrata API)

**Provider:** Texas Department of Public Safety
**Portal:** [data.texas.gov](https://data.texas.gov)
**Dataset ID:** `xvp6-c9hn`
**Endpoint:** `https://data.texas.gov/resource/xvp6-c9hn.csv`
**Granularity:** County × Year × Offense Type
**Coverage:** 2000 – present
**License:** Public Domain

### Key Columns

| Column | Description |
|---|---|
| `county` | Texas county name |
| `year` | Report year |
| `offense_type` | UCR offense category |
| `offense_count` | Number of reported offenses |

### Fetching

```python
from src.python.data.fetcher import fetch_texas_ucr_socrata

path = fetch_texas_ucr_socrata(limit=50000)
```

**Rate limits:** 1 000 rows/request without token; 50 000/request with a
free [Socrata app token](https://data.texas.gov/login).

---

## 2. FBI Crime Data Explorer (CDE)

**Provider:** Federal Bureau of Investigation
**API:** [api.usa.gov/crime/fbi/cde](https://api.usa.gov/crime/fbi/cde/)
**Endpoint:** `summarized/state/TX/all-offenses/{year_from}/{year_to}`
**Granularity:** State-level annual summaries
**Coverage:** 1979 – present
**License:** Public Domain

### Getting an API Key

1. Register at [api.data.gov/signup/](https://api.data.gov/signup/)
2. Use `DEMO_KEY` for testing (40 requests/hour limit)

### Fetching

```python
from src.python.data.fetcher import fetch_fbi_state_data

path = fetch_fbi_state_data(
    state_abbr="TX",
    year_from=2018,
    year_to=2022,
    api_key="YOUR_KEY",
)
```

---

## 3. US Census TIGER/Line Shapefiles

**Provider:** US Census Bureau
**Portal:** [census.gov/geo/maps-data/data/tiger-line.html](https://www.census.gov/geo/maps-data/data/tiger-line.html)
**Vintage:** 2022
**License:** Public Domain

### Files Used

| File | Description | Download Size |
|---|---|---|
| `tl_2022_us_county.zip` | All US counties | ~75 MB |
| `tl_2022_48_place.zip` | Texas cities/places | ~5 MB |

### Fetching

```python
from src.python.data.fetcher import (
    fetch_texas_counties_shapefile,
    fetch_texas_places_shapefile,
)

county_shp = fetch_texas_counties_shapefile()
places_shp = fetch_texas_places_shapefile()
```

Shapefiles are extracted to `data/shapefiles/`.

---

## 4. Synthetic Dataset (Demo / Testing)

A procedurally generated dataset is built into the package for
demonstrations and unit tests — no internet access required.

**Generator:** `src.python.data.loader.generate_synthetic_dataset()`

### Properties

| Property | Value |
|---|---|
| Points | Configurable (default 5 000) |
| Cities | 10 major Texas cities |
| Crime types | 10 UCR categories |
| Years | 2018–2022 |
| Distribution | Gaussian clusters around city centres |
| Severity | Weighted by crime type |

```python
from src.python.data.loader import generate_synthetic_dataset

gdf = generate_synthetic_dataset(n_incidents=10_000, seed=42)
```

---

## Data Schema (Canonical)

After preprocessing, all datasets conform to this schema:

| Column | Type | Description |
|---|---|---|
| `longitude` | float64 | WGS-84 longitude |
| `latitude` | float64 | WGS-84 latitude |
| `offense_type` | str | Normalised offense category |
| `year` | int | Incident year |
| `city` | str | City / place name (if available) |
| `county` | str | County name (after spatial join) |
| `severity` | int | Severity score 1–10 |
| `crime_category` | str | `violent` / `property` / `other` |
| `geometry` | Point | Shapely Point (EPSG:4326) |

---

## Data Privacy & Ethics

- All data used is **publicly available, aggregated** official crime statistics.
- No personal identifying information (PII) is stored or processed.
- The synthetic dataset is entirely procedurally generated.
- Users are responsible for complying with the terms of service of any
  external API they query.
