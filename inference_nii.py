import os
import torch

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

import numpy as np
import nibabel as nib
from torchvision.transforms import transforms

from model.generator import Generator

# NOTE: Set these before running inference; they control which checkpoint is loaded #
target_experiment = ''
best_epochs = {} # e.g., {'SIEMENS_to_Philips': 50, 'Philips_to_SIEMENS': 50}
#####################################################################################

# same preprocessing as training (expects inputs normalized to [-1, 1] range)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

def get_axial_index(nii:nib.Nifti1Image):
    """Return the index of the axial slice axis for the given NIfTI.

    The function inspects the affine orientation codes and returns the
    dimension index that corresponds to the superior/inferior axis. The
    result is used to split and re-stack the volume along the axial axis.
    """
    aff_code = np.array(nib.orientations.aff2axcodes(nii.affine))
    if 'S' in aff_code:
        idx = np.where(aff_code == 'S')[0][0]
    elif 'I' in aff_code:
        idx = np.where(aff_code == 'I')[0][0]
    else:
        # fallback: assume axial is the last axis
        idx = nii.ndim - 1
    return idx

def forward(model, x, transform=None):
    """Run a single 2D array through the model and return an 8-bit array.

    The model expects a normalized tensor; we apply `transform` to the
    provided 2D numpy array, run the forward pass, convert the output from
    [-1, 1] to [0, 255] and return a `uint8` image. If the output is
    nearly constant (std <= 0.01) we return a zeroed array to avoid
    saving uninformative slices.
    """
    # prepare tensor: (H, W) -> (1, 1, H, W)
    x = x.squeeze().unsqueeze(0).to(device, torch.float32)
    if transform is not None:
        x = transform(x)

    with torch.no_grad():
        x = model(x).detach().cpu().numpy().squeeze()

    # convert tanh output [-1,1] -> uint8 [0,255]
    x = (((x * 0.5) + 0.5) * 255.0).astype(np.uint8)
    if np.std(x) <= 0.01:
        return np.zeros_like(x)
    return x

def main(data_dir, save_dir, src_vendor='SIEMENS', trg_vendor='Philips'):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))])
    
    G = Generator(in_features=1, out_features=1, n_res_blocks=9)
    # load the trained generator checkpoint for the source->target vendor pair
    ckpt_path = os.path.join(
        'results', target_experiment, 'model',
        'netG_{}_to_{}_epoch_{}.pth'.format(
            src_vendor[0].upper(), trg_vendor[1].upper(),
            best_epochs['{}_to_{}'.format(src_vendor, trg_vendor)]
        )
    )
    G.load_state_dict(torch.load(ckpt_path))

    G = G.to(device)
    G.eval()

    os.makedirs(save_dir, exist_ok=True)

    nii_list = os.listdir(data_dir)
    for nii_path in nii_list:
        if nii_path.endswith('.nii') or nii_path.endswith('.nii.gz'):
            subj_id = nii_path.split('_')[0]

            fu_nii = nib.load(os.path.join(data_dir, nii_path))
            idx = get_axial_index(fu_nii)
            
            fu_arr = fu_nii.get_fdata()
            fu_arr = np.array_split(fu_arr, fu_arr.shape[idx], axis=idx)

            fu_harmonized_arr = [forward(G, arr, transform) for arr in fu_arr]
            
            fu_harmonized_arr = np.stack(fu_harmonized_arr, axis=idx)
            fu_harmonized_nii = nib.Nifti1Image(fu_harmonized_arr, fu_nii.affine, fu_nii.header)

            nib.save(fu_harmonized_nii, os.path.join(save_dir, f'{subj_id}_harmonized.nii.gz'))


if __name__=='__main__':
    data_dir = './dataset/Harmonization_dataset/testset/SIEMENS'
    save_dir = './results/{}/testset/SIEMENS_to_Philips'.format(target_experiment)
    # main(data_dir, save_dir, src_vendor='SIEMENS', trg_vendor='Philips')
    