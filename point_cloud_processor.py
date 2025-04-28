import numpy as np
import pandas as pd
from typing import Tuple, Optional

class PointCloudProcessor:
    def __init__(self, csv_file_path: str):
        """
        Initialize the PointCloudProcessor with a CSV file path.
        
        Args:
            csv_file_path (str): Path to the CSV file containing point cloud data
        """
        self.csv_file_path = csv_file_path
        self.point_clouds = None
        self.load_data()
    
    def load_data(self) -> None:
        """Load point cloud data from CSV file with specific format handling."""
        # Read the raw data, skipping:
        # - First row (Trajectories)
        # - Second row (100)
        # - Third row (Point names)
        # - Fourth row (X,Y,Z headers)
        # - Fifth row (units in mm)
        df = pd.read_csv(self.csv_file_path, skiprows=[0, 1, 2, 3, 4], header=None)
        
        # Drop the 'Frame' and 'Sub Frame' columns
        df = df.iloc[:, 2:]
        
        # Replace empty strings with NaN and convert to float
        df = df.replace('', np.nan).astype(float)
        
        # Get the values as numpy array
        data = df.values
        
        # Reshape the data into point clouds
        # The data has X,Y,Z columns for each point, so we need to reshape it
        num_frames = data.shape[0]  # Number of time steps
        num_points = data.shape[1] // 3  # Each point has X,Y,Z coordinates
        self.point_clouds = data.reshape(num_frames, num_points, 3)
    
    def get_point_cloud_at_time(self, time_idx: int) -> np.ndarray:
        """
        Get point cloud data for a specific time index.
        
        Args:
            time_idx (int): Time index to retrieve the point cloud
            
        Returns:
            np.ndarray: Point cloud data of shape (80, 3)
        """
        if self.point_clouds is None:
            raise ValueError("No point cloud data loaded")
        
        if time_idx < 0 or time_idx >= self.point_clouds.shape[0]:
            raise ValueError(f"Time index out of range. Should be between 0 and {self.point_clouds.shape[0]-1}")
            
        return self.point_clouds[time_idx]
    
    def compute_transformation(self, source_time: int, target_time: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute the homogeneous transformation matrix between two point clouds using Kabsch algorithm.
        Only considers common points (non-empty/non-NaN points).
        
        Args:
            source_time (int): Time index for the source point cloud
            target_time (int): Time index for the target point cloud
            
        Returns:
            Tuple containing:
            - Rotation matrix (3x3)
            - Translation vector (3x1)
            - RMSE alignment error
        """
        # Get point clouds
        source_pc = self.get_point_cloud_at_time(source_time)
        target_pc = self.get_point_cloud_at_time(target_time)
        
        # Find common points (non-NaN points)
        valid_points = ~(np.isnan(source_pc).any(axis=1) | np.isnan(target_pc).any(axis=1))
        source_points = source_pc[valid_points]
        target_points = target_pc[valid_points]
        
        if len(source_points) < 3:
            raise ValueError("Not enough common points to compute transformation")
        
        # Center the points
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Compute covariance matrix
        H = source_centered.T @ target_centered
        
        # Compute SVD
        U, _, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Handle special reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = target_centroid - R @ source_centroid
        
        # Compute alignment error (RMSE)
        transformed_source = (R @ source_points.T).T + t
        error = np.sqrt(np.mean(np.sum((transformed_source - target_points) ** 2, axis=1)))
        
        return R, t, error

    def get_homogeneous_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix and translation vector to 4x4 homogeneous transformation matrix.
        
        Args:
            R (np.ndarray): 3x3 rotation matrix
            t (np.ndarray): 3x1 translation vector
            
        Returns:
            np.ndarray: 4x4 homogeneous transformation matrix
        """
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t
        return H