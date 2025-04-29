import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import random


class PointCloudProcessor:
    """
    A CSV-based processor for time-varying point clouds.

    * Each frame stores **68** markers:
        - markers 0-2  → rigid dental support (optional)
        - markers 3-67 → anatomical points of interest (65 markers)

    By default the three support markers are **excluded** from every
    query; pass ``support=True`` when you wish to keep them.
    """

    # --------------------------------------------------------------------- #
    #  Construction / I/O                                                   #
    # --------------------------------------------------------------------- #

    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.point_clouds: Optional[np.ndarray] = None     # (T, 68, 3)
        self.load_data()

    def load_data(self) -> None:
        """Read the Motive/Qualisys-style CSV and reshape to (T, P, 3)."""
        df = pd.read_csv(
            self.csv_file_path,
            skiprows=[0, 1, 2, 3, 4],       # meta-headers
            header=None
        )

        # Drop 'Frame' + 'Sub Frame'
        df = df.iloc[:, 2:]

        # Replace empty strings, force float
        df = df.replace('', np.nan).astype(float)

        raw = df.values
        num_frames = raw.shape[0]
        num_points = raw.shape[1] // 3
        self.point_clouds = raw.reshape(num_frames, num_points, 3)

    # --------------------------------------------------------------------- #
    #  Internal helper                                                      #
    # --------------------------------------------------------------------- #

    def _slice_support(self, pc: np.ndarray, support: bool) -> np.ndarray:
        """Return view with / without the first three support markers."""
        return pc if support else pc[3:]

    # --------------------------------------------------------------------- #
    #  Public queries                                                       #
    # --------------------------------------------------------------------- #

    def get_point_cloud_at_time(
        self,
        time_idx: int,
        *,
        support: bool = False
    ) -> np.ndarray:
        """
        Retrieve a single frame.

        Parameters
        ----------
        time_idx : int
            Index of the desired frame.
        support : bool, default False
            Include the three reference markers (indices 0-2)?

        Returns
        -------
        pc : (65 or 68, 3) ndarray
        """
        if self.point_clouds is None:
            raise ValueError("No point cloud data loaded")

        if not (0 <= time_idx < self.point_clouds.shape[0]):
            raise ValueError(
                f"time_idx out of range (0 – {self.point_clouds.shape[0]-1})"
            )

        return self._slice_support(self.point_clouds[time_idx], support)

    def get_best_point_cloud(
        self,
        *,
        support: bool = False,
        rng: Optional[random.Random] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Return the densest frame (most complete XYZ triplets).

        When several frames tie, pick one at random.

        Returns
        -------
        best_pc  : ndarray
            The selected point cloud (support markers kept / removed
            according to *support*).
        best_idx : int
            Its frame index.
        """
        if self.point_clouds is None:
            raise ValueError("No point cloud data loaded")
        if rng is None:
            rng = random.Random()

        pcs = self.point_clouds if support else self.point_clouds[:, 3:, :]
        present_mask = ~np.isnan(pcs).any(axis=2)          # (T, P)
        counts = present_mask.sum(axis=1)                  # (T,)
        best_count = counts.max()
        candidates = np.where(counts == best_count)[0]

        best_idx = rng.choice(candidates)
        best_pc = pcs[best_idx]

        return best_pc, best_idx

    # --------------------------------------------------------------------- #
    #  Alignment                                                            #
    # --------------------------------------------------------------------- #

    def compute_transformation(
        self,
        source_time: int,
        target_time: int,
        *,
        support: bool = False,
        robust: bool = False,
        ransac_thresh: float = 5.0,
        max_iters: int = 1_000,
        min_inliers: int = 6,
        rng: Optional[random.Random] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
        """
        Estimate the rigid transform **between the two clouds' 65
        anatomical points by default**; set *support=True* to include
        the three reference markers as well.
        """
        if rng is None:
            rng = random.Random()

        src_pc = self.get_point_cloud_at_time(source_time, support=support)
        tgt_pc = self.get_point_cloud_at_time(target_time, support=support)

        valid = ~(np.isnan(src_pc).any(1) | np.isnan(tgt_pc).any(1))
        src, tgt = src_pc[valid], tgt_pc[valid]

        if src.shape[0] < 3:
            raise ValueError("Not enough common points")

        # Fast path – plain Kabsch
        if not robust:
            R, t, rmse = self._kabsch(src, tgt)
            return R, t, rmse, None

        # ---------------- RANSAC ---------------- #
        best_inliers: List[int] = []
        for _ in range(max_iters):
            sample = rng.sample(range(src.shape[0]), 3)
            R_s, t_s, _ = self._kabsch(src[sample], tgt[sample])

            pred = (R_s @ src.T).T + t_s
            dists = np.linalg.norm(pred - tgt, axis=1)
            inliers = np.where(dists < ransac_thresh)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = list(inliers)
                if len(best_inliers) >= min_inliers:
                    break

        if len(best_inliers) < 3:
            raise RuntimeError("RANSAC failed – insufficient inliers")

        R, t, rmse = self._kabsch(src[best_inliers], tgt[best_inliers])

        inlier_mask = np.zeros(src_pc.shape[0], dtype=bool)
        inlier_mask[np.where(valid)[0][best_inliers]] = True
        return R, t, rmse, inlier_mask

    # --------------------------------------------------------------------- #
    #  Utilities                                                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _kabsch(
        src: np.ndarray,
        tgt: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        src_c = src - src.mean(0)
        tgt_c = tgt - tgt.mean(0)
        U, _, Vt = np.linalg.svd(src_c.T @ tgt_c)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:       # reflection fix
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = tgt.mean(0) - R @ src.mean(0)
        rmse = np.sqrt(np.mean(np.sum(((R @ src.T).T + t - tgt) ** 2, axis=1)))
        return R, t, rmse

    @staticmethod
    def get_homogeneous_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Build a 4 × 4 homogeneous matrix from (R, t)."""
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        return H
