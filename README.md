# Automated Spicule Tracking and Parabolic Trajectory Analysis

## Overview

This repository contains a complete end-to-end pipeline for automatically detecting, tracking, and analyzing solar spicule trajectories from Pencil Code MHD simulations.

Starting from temperature snapshots stored in HDF5 format, the pipeline:

1. extracts candidate spicule regions,
2. detects spicule apex positions,
3. reconstructs individual tip trajectories,
4. classifies trajectories according to their parabolic quality,
5. fits ballistic trajectories,
6. extracts physical parameters such as initial velocity and acceleration.

The code was developed and tested on simulations with magnetic field strengths of **10 G** and **50 G**.

---

## Input Data

The pipeline expects HDF5 files containing:

* `temperature[frame,z,x]`
* `time[frame]`
* `x[x]`
* `z[z]`

Example datasets:

```text
data/raw/spicules_temperature_final.h5
data/raw/spicules_temperature_final_50G.h5
```

(The raw datasets are not tracked by Git because of their size.)

---

## Pipeline

### 1. Global normalization

The logarithmic temperature field

[
\log_{10}T(x,z,t)
]

is computed and normalized using a global background profile.

---

### 2. Temperature band segmentation

Spicule material is isolated using a temperature interval

[
T_{\min}\le \log_{10}T \le T_{\max}.
]

This produces a binary canopy mask.

---

### 3. Upper envelope extraction

For every horizontal column, the highest segmented pixel is found, yielding an upper envelope

[
z_{\rm env}(x).
]

---

### 4. Tip detection

Local maxima of the smoothed envelope are identified to obtain spicule tip locations

[
(x_i,z_i).
]

---

### 5. Multi-object tracking

Detected tips are linked across successive frames using:

* constant-velocity prediction,
* Hungarian assignment,
* birth/death handling,
* maximum distance constraints.

This reconstructs complete trajectories

[
{(x_i(t_k),z_i(t_k))}_{k=1}^{N_i}.
]

The improved tracker (`tracker_v2.py`) minimizes trajectory stitching and identity swaps.

---

### 6. Trajectory catalogs

Two data products are produced:

#### Full trajectory catalog

```text
outputs/*/tracks_full.json
```

contains complete time histories

```json
{
  "track_id": ...,
  "times": [...],
  "xs": [...],
  "zs": [...]
}
```

#### Summary catalog

```text
outputs/*/track_catalog.csv
```

contains trajectory statistics:

* lifetime,
* maximum height,
* horizontal drift,
* duration,
* missed frames.

---

### 7. Parabola classification

The vertical trajectories (z(t)) are automatically classified into five categories:

| Class | Description                           |
| ----- | ------------------------------------- |
| A     | Complete parabolas                    |
| B     | Incomplete but parabolic trajectories |
| C     | Endpoint distortions                  |
| D     | Kinky or imperfect parabolas          |
| E     | Non-parabolic trajectories            |

Class overview plots are stored in

```text
outputs/*/parabola_classification/
```

---

### 8. Quadratic fitting

Only classes A and B are fitted.

The model

[
z(t)=At^2+Bt+C
]

is fitted in physical units:

* height in km,
* time in seconds.

Physical quantities are extracted:

| Quantity               | Formula           |
| ---------------------- | ----------------- |
| Initial velocity       | (v_0=B)           |
| Signed acceleration    | (a=2A)            |
| Deceleration magnitude | (-2A)             |
| Apex time              | (-B/(2A))         |
| Apex height            | (z(t_{\rm apex})) |
| RMSE                   | fit residual      |
| (R^2)                  | goodness of fit   |

The final parameter catalog is stored in

```text
outputs/*/parabola_fits/
```

---

## Directory Structure

```text
src/
    tracker_v2.py
    normalize.py
    envelope.py
    ...

scripts/
    build_track_catalog_v2_full.py
    analyze_tracks.py
    classify_parabolas.py
    fit_parabolas.py
    ...

outputs/
    10G/
        analysis/
        movies/
        v2/
        parabola_classification/
        parabola_fits/

    50G/
        analysis/
        movies/
        v2/
        parabola_classification/
        parabola_fits/
```

---

## Main Data Product

The principal output of this repository is

[
\left{(x_i(t_k),z_i(t_k))\right}_{k=1}^{N_i}
]

for every reconstructed spicule.

These trajectories are stored in

```text
outputs/*/v2/tracks_full.json
```

and form the basis for subsequent physical analysis.

---

## Requirements

* Python ≥ 3.11
* NumPy
* SciPy
* Pandas
* Matplotlib
* h5py
* scikit-image

---

## Citation

If this repository contributes to published work, please cite the associated paper and reference the GitHub repository.
