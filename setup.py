"""
setup.py
========
Python package setup with optional C++ extension build via pybind11.

Install (pure Python, no C++ extension):
    pip install -e .

Install (with C++ extension):
    pip install -e ".[cpp]"
    # or build manually:
    # python setup.py build_ext --inplace

The C++ extension (texas_crime_spatial) is optional; the Python code falls
back to scikit-learn implementations when it is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import find_packages, setup

# ── Try to build the pybind11 extension ──────────────────────────────────────
ext_modules = []
cmdclass    = {}

try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "texas_crime_spatial",
            sources=[
                "src/cpp/spatial_index/kdtree.cpp",
                "src/cpp/clustering/dbscan.cpp",
                "src/cpp/bindings/spatial_ext.cpp",
            ],
            include_dirs=["src/cpp"],
            extra_compile_args=["-O3", "-std=c++17"],
            define_macros=[("VERSION_INFO", "1.0.0")],
            language="c++",
        )
    ]
    cmdclass = {"build_ext": build_ext}

except ImportError:
    print(
        "WARNING: pybind11 not found – C++ extension will not be built.\n"
        "  Run `pip install pybind11` then re-install to enable the C++ backend.",
        file=sys.stderr,
    )

# ── Read long description from README ────────────────────────────────────────
here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8") \
    if (here / "README.md").exists() else ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup(
    name="texas-crime-geospatial-analysis",
    version="1.0.0",
    author="Texas Crime Analysis Team",
    description="Geospatial crime pattern analysis and predictive mapping for Texas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/texas-crime-geospatial-analysis",
    license="MIT",
    python_requires=">=3.10",

    packages=find_packages(where=".", exclude=["tests*", "notebooks*", "docs*"]),
    package_dir={"": "."},

    ext_modules=ext_modules,
    cmdclass=cmdclass,

    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "geopandas>=0.14",
        "shapely>=2.0",
        "pyproj>=3.6",
        "fiona>=1.9",
        "libpysal>=4.9",
        "esda>=2.5",
        "folium>=0.15",
        "branca>=0.7",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "plotly>=5.18",
        "dash>=2.14",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "requests>=2.31",
        "PyYAML>=6.0",
        "pyarrow>=14.0",
    ],

    extras_require={
        "cpp":   ["pybind11>=2.11"],
        "dev":   ["pytest>=7.4", "pytest-cov>=4.1", "black>=23.0", "ruff>=0.1"],
        "notebook": ["jupyter>=1.0", "ipykernel>=6.0", "ipywidgets>=8.0"],
        "trend": ["pymannkendall>=1.4"],
    },

    entry_points={
        "console_scripts": [
            "texas-crime=main:main",
            "texas-fetch=scripts.fetch_data:main",
            "texas-analyse=scripts.run_analysis:main",
            "texas-maps=scripts.generate_maps:main",
        ]
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],

    include_package_data=True,
    zip_safe=False,
)
