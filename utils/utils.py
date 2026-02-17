import numpy as np
from skimage.metrics import structural_similarity as ssim

def ssim_batch(batch1, batch2, need_unnormalize=True):
    batch1 = batch1.detach().cpu().numpy()
    batch2 = batch2.detach().cpu().numpy()
    if need_unnormalize:
        batch1 = ((batch1 * 0.5) + 0.5) * 255.0
        batch2 = ((batch2 * 0.5) + 0.5) * 255.0
    ls = [ssim(batch1[i, :, :, :].squeeze().astype(np.uint8), batch2[i, :, :, :].squeeze().astype(np.uint8), data_range=255) for i in range(batch1.shape[0])] # Compute SSIM for each image pair in the batch
    return np.mean(ls)
    
def psnr_batch(batch1, batch2, need_unnormalize=True):
    batch1 = batch1.detach().cpu().numpy()
    batch2 = batch2.detach().cpu().numpy()
    if need_unnormalize:
        batch1 = ((batch1 * 0.5) + 0.5) * 255.0
        batch2 = ((batch2 * 0.5) + 0.5) * 255.0
    
    psnr_list = []
    for i in range(batch1.shape[0]):
        img1 = batch1[i, :, :, :].squeeze()
        img2 = batch2[i, :, :, :].squeeze()
        mse = np.mean((img1 - img2) ** 2)
        
        if mse == 0:
            psnr_value = np.inf # Infinite PSNR means the images are identical
        else:
            psnr_value = 10 * np.log10((2 ** 2) / mse)
        psnr_list.append(psnr_value)
    
    psnr_list = [x for x in psnr_list if not np.isinf(x)] # Filter out infinite PSNR values (identical images)
    return np.mean(psnr_list)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
