import torch
def call_cos_tensor(tensor1, tensor2):
    tensor1 = tensor1/torch.linalg.norm(tensor1,dim=1,keepdim=True)
    tensor2 = tensor2/torch.linalg.norm(tensor2,dim=1,keepdim=True)
    cosvalue = torch.sum(tensor1*tensor2,dim=1,keepdim=True)
    return cosvalue

def compute_perpendicular_component(latent_diff, latent_hat_uncond):
    shape = latent_diff.size()
    latent_diff = latent_diff.view(latent_diff.size(0), -1).float()
    latent_hat_uncond = latent_hat_uncond.view(latent_hat_uncond.size(0), -1).float()
    
    if latent_diff.size() != latent_hat_uncond.size():
        raise ValueError("latent_diff and latent_hat_uncond must have the same shape [n, d].")
    
    dot_product = torch.sum(latent_diff * latent_hat_uncond, dim=1, keepdim=True)  # [n, 1]
    norm_square = torch.sum(latent_hat_uncond * latent_hat_uncond, dim=1, keepdim=True)  # [n, 1]
    projection = (dot_product / (norm_square + 1e-8)) * latent_hat_uncond
    perpendicular_component = latent_diff - projection
    
    return projection.view(shape),perpendicular_component.view(shape)