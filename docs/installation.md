# Installation Guide

## Requirements

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 / 3.12 recommended |
| CMake | 3.18 | Required for C++ extension |
| C++ compiler | GCC 11 / Clang 14 / MSVC 2022 | C++17 support required |
| pip | 23+ | |

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/texas-crime-geospatial-analysis.git
cd texas-crime-geospatial-analysis
```

---

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux / macOS
.venv\Scripts\activate.bat      # Windows
```

---

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS note:** GeoPandas requires GDAL. The easiest path is Homebrew:
> ```bash
> brew install gdal
> pip install pygdal=="$(gdal-config --version).*"
> ```
> Or use `conda install -c conda-forge geopandas` instead.

> **Windows note:** Use the [OSGeo4W](https://trac.osgeo.org/osgeo4w/) distribution
> or install pre-compiled wheels from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

---

## 4. Build the C++ Extension (Optional but Recommended)

The C++ extension (`texas_crime_spatial`) provides a 10–50× speedup for
spatial queries. The Python code gracefully falls back to scikit-learn
when the extension is absent.

### Option A – via pip (recommended)

```bash
pip install -e ".[cpp]"
```

### Option B – via CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 4
cd ..
```

After either option, verify:

```python
import texas_crime_spatial as tcs
print(tcs.__doc__)
```

---

## 5. Install the Package (editable)

```bash
pip install -e .
```

This registers the `texas-crime`, `texas-fetch`, `texas-analyse`, and
`texas-maps` console commands.

---

## 6. Verify Installation

```bash
python main.py --mode demo
```

Expected output includes KDE bandwidth, DBSCAN cluster counts, and
confirmation that maps were saved to `outputs/maps/`.

---

## Conda Environment (Alternative)

A `conda` environment file is provided for users who prefer conda:

```bash
conda env create -f environment.yml
conda activate texas-crime
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: geopandas` | `pip install geopandas` or use conda |
| `ImportError: libgdal.so` | Install GDAL system library (see above) |
| `pybind11 not found` | `pip install pybind11` before CMake |
| CMake can't find Python | `cmake .. -DPython3_EXECUTABLE=$(which python3)` |
| Folium maps blank in browser | Use a modern Chromium-based browser |
| `OSError: cannot load shared library` (macOS) | `export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH` |
