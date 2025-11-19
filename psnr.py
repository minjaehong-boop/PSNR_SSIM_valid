import torch
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
import numpy as np

def off(preds, target):
    return peak_signal_noise_ratio(preds, target, data_range=1.0).item()

def val(preds, target):
    if not preds.is_floating_point():
        preds = preds.to(torch.float32)
    if not target.is_floating_point():
        target = target.to(torch.float32)
    
    sum_squared_error = torch.sum(torch.pow(preds - target, 2))
    num_obs = torch.tensor(target.numel(), device=target.device)
    data_range_val = torch.tensor(1.0)
    
    psnr_base_e = 2 * torch.log(data_range_val) - torch.log(sum_squared_error / num_obs)
    psnr_vals = psnr_base_e * (10 / torch.log(torch.tensor(10.0)))
    return psnr_vals.item()

if __name__ == "__main__":
    gt_data = np.loadtxt('gt.csv', delimiter=',')
    pr_data = np.loadtxt('pr.csv', delimiter=',')
    
    gt_tensor = torch.from_numpy(gt_data).float().unsqueeze(0).unsqueeze(0)
    pr_tensor = torch.from_numpy(pr_data).float().unsqueeze(0).unsqueeze(0)
    
    psnr_off = off(pr_tensor, gt_tensor)
    psnr_val = val(pr_tensor, gt_tensor)

    print(f"PSNR Score (Official): {psnr_off:.4f}")
    print(f"PSNR Score (Validation): {psnr_val:.4f}")