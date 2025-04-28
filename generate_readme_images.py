import os
import numpy as np
from point_cloud_processor import PointCloudProcessor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(points, ax, color='b', title=None, alpha=1.0):
    """Helper function to visualize a point cloud"""
    valid_points = ~np.isnan(points).any(axis=1)
    points = points[valid_points]
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker='o', s=50, alpha=alpha)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Add grid lines
    ax.grid(True)
    
    # Set consistent axis limits
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([1200, 1400])
    
    if title:
        ax.set_title(title)
    return scatter

def main():
    # Create images directory if it doesn't exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # Get the first CSV file from the sample_data directory
    sample_file = os.path.join('sample_data', '20241211_MR_KT_Statique.csv')
    processor = PointCloudProcessor(sample_file)
    
    # Get point clouds
    pc1 = processor.get_point_cloud_at_time(0)
    pc2 = processor.get_point_cloud_at_time(1)
    
    # Compute transformation
    R, t, error = processor.compute_transformation(0, 1)
    transformed_points = (R @ pc1.T).T + t

    # 1. Create comparison visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Source point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = visualize_point_cloud(pc1, ax1, 'b', 'Source Point Cloud')
    ax1.view_init(elev=20, azim=45)
    
    # Target point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = visualize_point_cloud(pc2, ax2, 'r', 'Target Point Cloud')
    ax2.view_init(elev=20, azim=45)
    
    # Overlay of transformed and target
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = visualize_point_cloud(transformed_points, ax3, 'g', 'Alignment Result')
    scatter4 = visualize_point_cloud(pc2, ax3, 'r', alpha=0.5)
    ax3.view_init(elev=20, azim=45)
    ax3.legend([scatter3, scatter4], ['Transformed Source', 'Target'])
    
    # Add a super title
    plt.suptitle(f'Point Cloud Alignment (RMSE: {error:.3f} mm)', y=1.08)
    
    plt.tight_layout()
    plt.savefig('images/point_cloud_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Update README with images
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Add image after the Features section
    image_text = "\n\n![Point Cloud Processing Example](images/point_cloud_comparison.png)\n"
    features_end = content.find("## Requirements")
    if features_end != -1:
        content = content[:features_end] + image_text + content[features_end:]
        
        with open('README.md', 'w') as f:
            f.write(content)

if __name__ == "__main__":
    main()