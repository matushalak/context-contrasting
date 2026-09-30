from torch.utils.data import Dataset
import torch
from torch.nn.functional import grid_sample
import torchvision
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R
import os

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from augmentations import *
import copy
from pathlib import Path

from torchvision import datasets, transforms


############################# STL10 SalMap dataset #############################
### Download STL10 images from https://cs.stanford.edu/~acoates/stl10/
### Download STL10 saliency maps from hugging face https://huggingface.co/datasets/Hafez/salmap-stl10
class STL10_SalMap(Dataset):
    def __init__(self, data_path, sal_path, split, num_patches, use_sal, ior):
        """
            data_path (string): Path to the directory containing the dataset files.
            sal_path (string): Path to the directory containing the saliency maps.
            split (string): One of {'train', 'test', 'unlabeled'} to specify the dataset split.
            num_patches (int): Number of patches to sample from the image.
            use_sal (bool): Whether to use saliency map for sampling patch centers.
            ior (bool): Whether to use inhibition of return when sampling patch centers.
        """
        self.dataset = torchvision.datasets.STL10(root=data_path, split=split, download=False)
        self.sal_path = os.path.join(sal_path, split)
        if split == 'train':
            self.sal_path = os.path.join(self.sal_path, 'saliency_train.npy')
        elif split == 'test':
            self.sal_path = os.path.join(self.sal_path, 'saliency_test.npy')
        elif split == 'unlabeled':
            self.sal_path = os.path.join(self.sal_path, 'saliency_unlabeled.npy')
        else:
            raise ValueError("Invalid split")
        self.saliency_maps = np.load(self.sal_path)
        assert len(self.dataset) == len(self.saliency_maps), "Mismatch between dataset size and saliency maps"
        self.use_sal = use_sal
        self.ior = ior
        self.num_patches = num_patches
        
        norm_params = {"mean": [0.4467, 0.4398, 0.4066], "std": [0.2241, 0.2215, 0.2239],}
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),])
        self.full_img_trans = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(), transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"]),])
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        sal_map = self.saliency_maps[idx]    # Shape: (1, 96, 96)
        
        sal_map = torch.from_numpy(sal_map).float()
        if len(sal_map.shape) == 2:
            sal_map = sal_map.unsqueeze(0)
        
        sac_positions = None
        patches = None

        rgb_image = image
        image_size = 96
        fovea_size = 32
        sac_positions = []
        inh_radius = fovea_size//2
        trans_hflip = transforms.functional.hflip
        trans_rot = transforms.functional.rotate
        img_full = transforms.ToTensor()(rgb_image)
        img_full = img_full.unsqueeze(0)
        probs_inh = copy.deepcopy(sal_map)
        probs_inh = probs_inh.unsqueeze(0)
        for _ in range(self.num_patches):
            if self.use_sal:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                sac_position = torch.multinomial(probs_flat, num_samples=1)
                sac_positions.append(sac_position)
            else:
                probs_flat = probs_inh.view(probs_inh.size(0), -1)
                num_elements = probs_flat.size(1)  # Number of elements in each row of probs_flat
                uniform_value = 1.0 / num_elements
                uniform_dist_tensor = torch.full(probs_flat.size(), uniform_value)
                sac_position = torch.multinomial(uniform_dist_tensor, num_samples=1)
                sac_positions.append(sac_position)
            # Calculate row and column from sac_position
            row = sac_position // image_size
            col = sac_position % image_size
            # Create inhibition mask
            if self.ior:
                rows = torch.arange(probs_inh.size(2)).view(1, 1, -1, 1)
                cols = torch.arange(probs_inh.size(3)).view(1, 1, 1, -1)
                row_mask = (rows >= (row - inh_radius).unsqueeze(2)) & (rows <= (row + inh_radius).unsqueeze(2))
                col_mask = (cols >= (col - inh_radius).unsqueeze(2)) & (cols <= (col + inh_radius).unsqueeze(2))
                inh_mask = row_mask & col_mask
                inh_mask = inh_mask.permute(1, 0, 2, 3) 
                probs_inh = probs_inh * (1 - inh_mask.float()) + 1e-16
                probs_inh = probs_inh / probs_inh.sum(dim=(2, 3), keepdim=True)
        # Concatenate and shuffle the order of saccade positions
        sac_positions = torch.cat(sac_positions, dim=1)

        ### Get patches
        rows = sac_positions // image_size
        cols = sac_positions % image_size
        centers = torch.stack((rows, cols), dim=2)
        
        grid_centers = centers.reshape(-1, 2)
        grid = generate_cropping_grid(image_size, fovea_size, grid_centers, "cpu")
        repeated_batch = img_full.unsqueeze(1).repeat(1, self.num_patches, 1, 1, 1)
        repeated_batch = repeated_batch.reshape(-1, 3, image_size, image_size)
        foveated_x = grid_sample(repeated_batch, grid, mode='bilinear', padding_mode='zeros')
        foveated_x = trans_rot(foveated_x,-90)
        foveated_x = trans_hflip(foveated_x)
        foveated_x = foveated_x.reshape((-1,3,fovea_size,fovea_size))
        foveated_x_ = foveated_x.reshape((self.num_patches, 3, fovea_size, fovea_size))
        
        sac_pos_norm = centers / image_size
        sac_pos_norm_reshaped = sac_pos_norm.reshape((self.num_patches, 2))
        to_pil = transforms.ToPILImage()
        patches = []
        for img_tensor in foveated_x_:
            # Convert tensor to PIL Image
            img_pil = to_pil(img_tensor)
            patch = self.transform(img_pil)
            patches.append(patch)
        patches = torch.stack(patches, dim=0)
        rgb_data = self.full_img_trans(rgb_image)
        sac_positions = sac_pos_norm_reshaped

        return rgb_data, sal_map, label, patches, sac_positions


#### UTILS
def generate_cropping_grid(input_size, crop_size, centers, device):
    # Create normalized grid
    ys = torch.linspace(-1, 1, crop_size, device=device)
    xs = torch.linspace(-1, 1, crop_size, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys)

    # Convert centers to normalized coordinates
    centers = torch.tensor(centers, device=device).float()
    centers = (centers * 2) / torch.tensor([input_size, input_size], device=device) - 1

    # Adjust grid using broadcasting
    grid_x = grid_x.unsqueeze(0) * (crop_size / input_size) + centers[:, 0].unsqueeze(-1).unsqueeze(-1)
    grid_y = grid_y.unsqueeze(0) * (crop_size / input_size) + centers[:, 1].unsqueeze(-1).unsqueeze(-1)

    grid = torch.stack([grid_x, grid_y], -1)
    return grid