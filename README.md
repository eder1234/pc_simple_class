# Point Cloud Processing Tool

A lightweight Python toolkit for analysing **time‑series 3‑D point‑clouds** exported by motion‑capture systems (Qualisys, Motive, etc.).  It provides high‑level helpers for loading CSV files, selecting frames, and computing frame‑to‑frame rigid registration – now with **robust outlier rejection** and an option to **ignore or keep three reference‑frame markers** that are sometimes present in dental / cranio‑facial studies.

---

## ✨  Key Features

| Category | Capability |
| -------- | ---------- |
| Loading  | One‑line import from vendor CSV; auto‑reshapes to `(frames × points × 3)` |
| Selection | `get_point_cloud_at_time(idx, support=False)` – fetch any frame, with or without the first three support markers |
|          | `get_best_point_cloud(support=False)` – pick the frame that has the most complete XYZ triplets (ties broken at random) |
| Registration | `compute_transformation(src, tgt, support=False, robust=False)` – Kabsch alignment with optional **RANSAC** outlier rejection |
| Metrics   | RMSE and full inlier mask returned by the alignment routine |
| Visuals   | Helper routines for 3‑D scatter / overlay plots (see `visualise.py`) |

---

## 📦  Requirements

* Python ≥ 3.9
* NumPy
* Pandas
* Matplotlib

Install with pip:

```bash
pip install -r requirements.txt
```

or via Conda:

```bash
conda env create -f environment.yml
```

---
## 🚀  Quick Start

```python
from point_cloud_processor import PointCloudProcessor

# 1. Load the CSV
pcp = PointCloudProcessor("data/subject01.csv")

# 2. Grab two frames **without** the three dental‑support markers
cloud_a = pcp.get_point_cloud_at_time(0, support=False)
cloud_b = pcp.get_point_cloud_at_time(42, support=False)

# 3. Robust alignment (RANSAC) on the 65 anatomical points
R, t, rmse, inliers = pcp.compute_transformation(0, 42, support=False, robust=True)
print(f"RMSE = {rmse:.2f} mm  –  {inliers.sum()} inliers")

# 4. Need the densest frame? – Easy:
best_cloud, best_idx = pcp.get_best_point_cloud()
print(f"Frame {best_idx} has the most complete data → {best_cloud.shape[0]} points")
```

### Data Format

The loader expects the standard Qualisys CSV layout:

1. **Row 1**  `Trajectories`
2. **Row 2**  number of frames
3. **Row 3**  marker names
4. **Row 4**  coordinate headers (`..., X, Y, Z, ...`)
5. **Row 5**  units (`mm`)
6. **Rows 6+**  numeric data

> **Marker count**: the file may contain **68 markers** – the first three (`0‑2`) are an *optional* rigid dental support.  Most analyses set `support=False` so that all methods operate on the **65 anatomical markers** only.

---
## 🔧  Implementation Notes

* **Transformation** – classic Kabsch (centroid alignment + SVD) plus an optional RANSAC wrapper (configurable threshold / iterations).
* **Outlier Mask** – when `robust=True`, the boolean mask returned lets you visualise or further process the inliers.
* **API Stability** – all new flags default to the previous behaviour, so existing scripts run unchanged.

---

## 🖼️  Examples & Visualisation

![Registered point clouds](images/point_cloud_comparison.png)

See `demo.py` for a self‑contained walkthrough that loads a CSV, aligns two frames, and plots the result.

```bash
python demo.py
```

---
## 🤝  Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-awesome`)  
3. Commit your changes (+ tests!)
4. Open a pull request

---
