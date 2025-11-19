import torch
from torch.nn import functional as F
from torchmetrics.functional.image.ssim import structural_similarity_index_measure
import numpy as np

def off(preds, target):
    return structural_similarity_index_measure(
        preds, target, data_range=1.0, gaussian_kernel=False, kernel_size=3, k1=0.01, k2=0.03
    ).item()

def val(preds, target):


    kernel_size = [3, 3]
    sigma = [1.5, 1.5]
    data_range = 1.0
    k1 = 0.01
    k2 = 0.03

    c1 = pow(k1 * data_range, 2)
    #print(c1)
    c2 = pow(k2 * data_range, 2)
    #print(c2)
    device = preds.device
    channel = preds.size(1)
    dtype = preds.dtype

    
    kernel = torch.ones((channel, 1, *kernel_size), dtype=dtype, device=device) / torch.prod(
        torch.tensor(kernel_size, dtype=dtype, device=device)
    )
    pad_h = (kernel_size[0] - 1) // 2
    pad_w = (kernel_size[1] - 1) // 2
    
    preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    input_list = torch.cat((preds, target, preds * preds, target * target, preds * target))
    outputs = F.conv2d(input_list, kernel, groups=channel)
    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target.to(dtype) + c2
    lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    
    return ssim_idx_full_image.reshape(ssim_idx_full_image.shape[0], -1).mean(-1).item()

if __name__ == "__main__":
    gt_data = np.loadtxt('gt.csv', delimiter=',')
    pr_data = np.loadtxt('pr.csv', delimiter=',')
    
    gt_tensor = torch.from_numpy(gt_data).float().unsqueeze(0).unsqueeze(0)
    pr_tensor = torch.from_numpy(pr_data).float().unsqueeze(0).unsqueeze(0)
    
    ssim_off = off(pr_tensor, gt_tensor)
    ssim_val = val(pr_tensor, gt_tensor)
    
    print(f"SSIM Score (Official): {ssim_off:.4f}")
    print(f"SSIM Score (Validation): {ssim_val:.4f}")