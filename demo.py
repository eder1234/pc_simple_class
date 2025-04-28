import os
import numpy as np
from point_cloud_processor import PointCloudProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points, ax, color='b', title=None, alpha=1.0):
    """Helper function to visualize a point cloud"""
    valid_points = ~np.isnan(points).any(axis=1)
    points = points[valid_points]
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='o', s=50, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

def main():
    # Get the first CSV file from the sample_data directory
    sample_file = os.path.join('sample_data', '20241211_MR_KT_Statique.csv')
    
    # Create a PointCloudProcessor instance
    processor = PointCloudProcessor(sample_file)
    
    # 1. Demonstrate getting point clouds at different times
    pc1 = processor.get_point_cloud_at_time(0)  # First frame
    pc2 = processor.get_point_cloud_at_time(1)  # Second frame
    
    print(f"Shape of point cloud: {pc1.shape}")
    print(f"Number of valid points in first frame: {(~np.isnan(pc1).any(axis=1)).sum()}")
    
    # 2. Compute transformation between two point clouds
    try:
        R, t, error = processor.compute_transformation(0, 1)
        print("\nTransformation Results:")
        print("Rotation matrix:")
        print(R)
        print("\nTranslation vector:")
        print(t)
        print(f"\nAlignment error (RMSE): {error:.6f}")
        
        # Get homogeneous transformation matrix
        H = processor.get_homogeneous_matrix(R, t)
        print("\nHomogeneous Transformation Matrix:")
        print(H)
        
        # 3. Visualize the point clouds
        fig = plt.figure(figsize=(15, 5))
        
        # Original point clouds
        ax1 = fig.add_subplot(131, projection='3d')
        visualize_point_cloud(pc1, ax1, 'b', 'Source Point Cloud')
        
        ax2 = fig.add_subplot(132, projection='3d')
        visualize_point_cloud(pc2, ax2, 'r', 'Target Point Cloud')
        
        # Transformed point cloud
        ax3 = fig.add_subplot(133, projection='3d')
        # Transform pc1 using the computed transformation
        transformed_points = (R @ pc1.T).T + t
        visualize_point_cloud(transformed_points, ax3, 'g', 'Transformed Source Cloud')
        visualize_point_cloud(pc2, ax3, 'r', alpha=0.5)  # Overlay target cloud
        
        plt.tight_layout()
        plt.show()
        
    except ValueError as e:
        print(f"Error during transformation: {e}")

if __name__ == "__main__":
    main()