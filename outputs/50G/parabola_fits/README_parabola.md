# Parabolic Classification and Fitting of Spicule Trajectories

## Overview

After reconstructing individual spicule tip trajectories from the simulation data, the final stage of the pipeline is to determine whether each trajectory follows an approximately ballistic (parabolic) evolution and, if so, to extract physically meaningful kinematic parameters.

The input to this stage is the file

```
outputs/v2/tracks_full.json
```

which contains the reconstructed tip coordinates for every tracked spicule. Each entry stores the complete time history

$$
\left\{ (x_i(t_k),z_i(t_k)) \right\}_{k=1}^{N_i},
$$

where

* $x_i,t_k$ = horizontal tip position,
* $z_i,t_k$ = vertical (apex) tip position,
* $t_k$ = simulation time,
* $N_i$ = number of detections belonging to trajectory (i).

The objective of this stage is:

1. Classify trajectories according to how closely they resemble a parabola.
2. Reject obviously non-parabolic trajectories.
3. Fit a quadratic model to acceptable trajectories.
4. Extract physical parameters such as initial velocity and acceleration.
5. Quantify fit quality using standard statistical metrics.

---

# 1. Input Data

Each entry in `tracks_full.json` has the form

```json
{
  "track_id": 134,
  "times": [...],
  "xs": [...],
  "zs": [...]
}
```

Only the `times` and `zs` arrays are required for the parabolic analysis.

For each track, the fitting variable is defined as

$$
\tau = t - t_0,
$$

where $t_0$ is the birth time of the trajectory.

Using the shifted time coordinate has two advantages:

* it avoids numerical instability associated with fitting large absolute times,
* the fitted linear coefficient directly corresponds to the initial velocity.

Thus, every trajectory is represented as

$$
{(\tau_k, z_k)}_{k=1}^{N_i},
$$

with

$$
\tau_0 = 0.
$$

---

# 2. Quadratic Model

Each candidate trajectory is approximated by

$$
z(\tau) = A\tau^2 + B\tau + C.
$$

The coefficients are obtained using a standard least-squares quadratic fit via

```python
numpy.polyfit$tau, z, deg=2$
```

which minimizes

$$
\sum_k \left[z_k - (A\tau_k^2+B\tau_k+C)\right]^2.
$$

The fitted model is then evaluated against the observed data.

---

# 3. Physical Interpretation

Since

$$
z(\tau)=A\tau^2+B\tau+C,
$$

the velocity is

$$
v(\tau)=\frac{dz}{d\tau}=2A\tau+B,
$$

and the acceleration is

$$
a(\tau)=\frac{d^2z}{d\tau^2}=2A.
$$

The following physical quantities are computed for every accepted fit.

| Quantity               | Formula                                          | Description                                             |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| Initial velocity       | $v_0=B$                                          | Upward velocity at birth                                |
| Signed acceleration    | $a=2A$                                           | Physical acceleration (typically negative)              |
| Deceleration magnitude | $-2A$                                            | Positive quantity commonly quoted in spicule literature |
| Apex time              | $t_{\rm apex}=-\frac{B}{2A}$                     | Time at which fitted velocity becomes zero              |
| Fitted apex height     | $z_{\rm apex}=A t_{\rm apex}^2+B t_{\rm apex}+C$ | Maximum height predicted by the quadratic model         |

Both the signed acceleration and positive deceleration magnitude are stored in the final catalog to avoid ambiguity.

---

# 4. Trajectory Classification

Visual inspection of the reconstructed trajectories revealed that not all (z(t)) curves are equally suitable for direct fitting. The trajectories naturally fall into five categories.

## Class A: Complete Parabolas

Characteristics:

* clear rise and fall,
* single smooth apex,
* no visible distortions,
* complete trajectory observed.

These are the highest-quality trajectories and are ideal candidates for fitting.

---

## Class B: Partial Parabolas

Characteristics:

* only ascending branch or descending branch observed,
* trajectory truncated because tracking starts late or ends early,
* still locally consistent with a parabola.

Although incomplete, these trajectories can often still be fit successfully.

---

## Class C: Endpoint Artefacts

Characteristics:

* central portion follows a parabola,
* one or both ends contain small tracking artefacts,
* distortions likely arise from birth/death assignment uncertainty.

The central ballistic portion is generally still recoverable.

---

## Class D: Imperfect / Kinky Parabolas

Characteristics:

* overall parabolic envelope,
* local dents, kinks, or small deviations,
* moderate tracking noise or physical perturbations.

These trajectories are retained if the overall fit quality remains acceptable.

---

## Class E: Rejected Trajectories

Characteristics:

* no obvious parabolic structure,
* multiple turning points,
* oscillatory or irregular behaviour,
* likely tracking failures or overlapping structures.

These trajectories are excluded from the physical analysis.

---

# 5. Automatic Classification Procedure

The classification algorithm computes several diagnostic quantities for each trajectory:

* lifetime,
* number of turning points,
* quadratic fit $R^2$,
* normalized RMSE,
* whether the fitted apex lies inside the observed interval,
* endpoint residual behaviour.

Based on these diagnostics, each trajectory is assigned to one of the five classes described above.

The resulting plots are written to

```
outputs/parabola_classification/
```

including:

* `class_A.png`
* `class_B.png`
* `class_C.png`
* `class_D.png`
* `class_E.png`

where all trajectories of a given class are plotted together for visual verification.

---

# 6. Fit Quality Metrics

Two standard goodness-of-fit metrics are computed.

## Coefficient of Determination ($R^2$)

$$
\frac{
\sum_k (z_k-z_{\rm fit,k})^2
}{
\sum_k (z_k-\bar z)^2
}
$$

Interpretation:

* $R^2=1$: perfect fit,
* $R^2\approx0.95-1.00$: excellent agreement,
* lower values indicate increasing deviation from a quadratic trajectory.

---

## Root Mean Square Error (RMSE)

$$
{\rm RMSE}
=

\sqrt{
\frac{1}{N}
\sum_k
(z_k-z_{\rm fit,k})^2
}.
$$

RMSE measures the typical vertical deviation (in Mm) between the observed trajectory and the fitted parabola.

A normalized RMSE (NRMSE) is also used during classification:

$$
{\rm NRMSE}
=

\frac{{\rm RMSE}}
{z_{\max}-z_{\min}}.
$$

This makes the fit criterion independent of the overall trajectory amplitude.

---

# 7. Acceptance Criteria

Only trajectories satisfying all quality requirements are retained for the final physical analysis.

Typical acceptance conditions include:

* classification label is not "rejected",
* minimum number of observed points,
* finite quadratic coefficients,
* $R^2$ above a chosen threshold,
* RMSE below a chosen threshold,
* fitted apex lies reasonably close to the observed time interval.

Trajectories failing these criteria are stored separately but are excluded from parameter statistics.

---

# 8. Output Files

## 8.1 Classification Outputs

Directory:

```
outputs/parabola_classification/
```

Contains:

* class-by-class overview figures,
* summary statistics,
* trajectory classification catalog.

---

## 8.2 Parabola Fit Outputs

Directory:

```
outputs/parabola_fits/
```

Contains:

* accepted trajectory overlays,
* rejected trajectory overlays,
* histogram of initial velocities,
* histogram of accelerations,
* histogram of fitted apex heights,
* histogram of $R^2$,
* histogram of RMSE,
* CSV/JSON catalog of fitted parameters.

---

## 8.3 Per-Track Stored Parameters

For every accepted trajectory, the output catalog stores quantities such as:

| Field          | Description                             |
| -------------- | --------------------------------------- |
| `track_id`     | Unique trajectory identifier            |
| `A`, `B`, `C`  | Quadratic coefficients                  |
| `v0`           | Initial velocity                        |
| `acceleration` | Signed acceleration ($2A$)              |
| `deceleration` | Positive deceleration magnitude ($-2A$) |
| `t_apex`       | Fitted apex time                        |
| `z_apex`       | Fitted apex height                      |
| `r2`           | Coefficient of determination            |
| `rmse`         | Root mean square fitting error          |
| `class_label`  | Parabola classification class           |

---

# 9. Interpretation

The fitting stage is not intended to prove that every spicule follows exact ballistic motion. Instead, it provides:

* a quantitative measure of how closely reconstructed trajectories resemble parabolic evolution,
* a systematic method for rejecting poor or ambiguous tracks,
* a reproducible way of extracting kinematic parameters from high-quality trajectories.

The classification step is particularly important because it prevents obviously non-ballistic or poorly reconstructed trajectories from biasing the derived physical parameters.

---

# 10. Pipeline Summary

The complete trajectory-analysis workflow is therefore:

```
tracks_full.json
        │
        ▼
Extract (t, z) trajectories
        │
        ▼
Shift to local time τ = t - t0
        │
        ▼
Quadratic least-squares fit
        │
        ▼
Compute R², RMSE, turning points, endpoint diagnostics
        │
        ▼
Assign Class A / B / C / D / E
        │
        ├──────────► Reject Class E
        │
        ▼
Accept high-quality trajectories
        │
        ▼
Extract:
    • initial velocity
    • acceleration / deceleration
    • apex time
    • apex height
    • fit-quality metrics
        │
        ▼
Generate plots and final parameter catalog
```

This stage constitutes the final scientific analysis layer built on top of the automated spicule tracking pipeline.
