import matplotlib.pyplot as plt
import numpy as np
import torch
from psnr import off as psnr_off, val as psnr_val
from ssim import off as ssim_off, val as ssim_val

def draw_comparison(gt_path, pr_path, output_path='comparison.png'):
    """
    Loads ground truth and prediction data, calculates PSNR and SSIM,
    and saves a side-by-side comparison image.
    """
    # Load data
    gt_data = np.loadtxt(gt_path, delimiter=',')
    pr_data = np.loadtxt(pr_path, delimiter=',')
    
    gt_tensor = torch.from_numpy(gt_data).float().unsqueeze(0).unsqueeze(0)
    pr_tensor = torch.from_numpy(pr_data).float().unsqueeze(0).unsqueeze(0)

    # Calculate metrics
    psnr_official = psnr_off(pr_tensor, gt_tensor)
    psnr_validation = psnr_val(pr_tensor, gt_tensor)
    ssim_official = ssim_off(pr_tensor, gt_tensor)
    ssim_validation = ssim_val(pr_tensor, gt_tensor)

    # Convert tensors to numpy arrays for visualization
    gt_image = gt_tensor.squeeze().numpy()
    pr_image = pr_tensor.squeeze().numpy()

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot Ground Truth
    im1 = axes[0].imshow(gt_image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Plot Prediction
    im2 = axes[1].imshow(pr_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Add a colorbar
    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.5)

    # Add a title with the scores
    title = (
        f"PSNR (Official): {psnr_official:.4f} | PSNR (Validation): {psnr_validation:.4f}\n"
        f"SSIM (Official): {ssim_official:.4f} | SSIM (Validation): {ssim_validation:.4f}"
    )
    fig.suptitle(title, fontsize=12)

    # Save the figure
    plt.savefig(output_path)
    print(f"Comparison image saved to {output_path}")

if __name__ == '__main__':
    draw_comparison('gt.csv', 'pr.csv')
