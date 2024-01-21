import os.path
from pathlib import Path
import cv2
import torch
import numpy as np
import imgaug
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pickle
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# from utils import gray2rgb, normalize, overlay


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params, num_nodes=5):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
        self.num_nodes = num_nodes
        self.neighbourhood = pickle.load(
            open(
                '/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/min_distance_neighbouring/patient_with_5_neighbour',
                'rb'))

    def extract_files(self, root):
        """
        Extract the paths to all slices given the root path (ends with train or val)
        """
        files = []
        Index = {}
        for subject in root.glob("*"):  # Iterate over the subjects
            slice_path = subject / "ct"  # Get the slices for current subject
            patient = subject.parts[-1]
            start_num = len(files)

            for slice in slice_path.glob("*"):
                files.append(slice)

            end_num = len(files)
            Index[patient.strip('\n')] = np.array([start_num, end_num])
        self.Index = Index
        return files

    @staticmethod
    def change_img_to_label_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "mask"
        return Path(*parts)

    @staticmethod
    def change_img_to_pet_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "pet"
        return Path(*parts)

    @staticmethod
    def change_img_to_suv_path(path):
        """
        Replace data with mask to get the masks
        """
        parts = list(path.parts)
        parts[parts.index("ct")] = "suv"
        return Path(*parts)

    def augment(self, cts, pets, masks):
        """
        Augments slice and segmentation mask in the exact same way
        Note the manual seed initialization
        """
        ###################IMPORTANT###################
        # Fix for https://discuss.pytorch.org/t/dataloader-workers-generate-the-same-random-augmentations/28830/2
        random_seed = torch.randint(0, 1000000, (1,))[0].item()
        imgaug.seed(random_seed)
        #####################################################
        self.augment_params = self.augment_params.to_deterministic()
        mask_augs = []
        ct_augs = []
        pet_augs = []
        for i in range(self.num_nodes):
            masks[i] = SegmentationMapsOnImage(masks[i], masks[i].shape)
            ct_aug, mask_aug = self.augment_params(image=cts[i], segmentation_maps=masks[i])
            ct_augs.append(ct_aug)
            mask_augs.append(mask_aug)

            pet_aug, mask_aug = self.augment_params(image=pets[i], segmentation_maps=masks[i])
            pet_augs.append(pet_aug)

            mask_augs[i] = mask_augs[i].get_arr()

        return ct_augs, pet_augs, mask_augs

    @staticmethod
    def normalize(img):
        min = np.min(img)
        max = np.max(img)
        img = img - (min)
        img = img / (max - min)
        return img

    def __len__(self):
        """
        Return the length of the dataset (length of all files)
        """
        return len(self.all_files)

    def __getitem__(self, idx_base):
        cts = []
        suvs = []
        masks = []

        root = '/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train'
        file_path = self.all_files[idx_base]
        patient = self.all_files[idx_base].parts[-3]
        mask_name = file_path.parts[-1]

        target_mask_path = self.change_img_to_label_path(file_path)
        target_suv_path = self.change_img_to_suv_path(file_path)
        target_ct = (np.load(file_path))
        target_mask = np.load(target_mask_path)

        cts.append(target_ct / 3071)
        masks.append(target_mask)
        suvs.append(np.load(target_suv_path))
        # if (np.any(target_mask)):
        #     target_ct = normalize(target_ct) * 255
        #     target_mask = (target_mask * 255).astype(np.uint8)
        #     target_gt_overlay = overlay(gray2rgb(target_ct), target_mask, (0, 255, 0), 0.3)
        #     plt.imshow(target_gt_overlay)
        #     plt.show()
        if np.any(target_mask):
            for i in range(self.num_nodes - 1):
                i_neighb = self.neighbourhood[patient][mask_name][i]
                i_neighb_addr_ct = os.path.join(root, i_neighb['search_p'], 'ct', i_neighb['search_mask_name'])
                i_neighb_addr_suv = os.path.join(root, i_neighb['search_p'], 'suv', i_neighb['search_mask_name'])
                i_neighb_addr_mask = os.path.join(root, i_neighb['search_p'], 'mask', i_neighb['search_mask_name'])

                i_ct = np.load(i_neighb_addr_ct)
                i_mask = np.load(i_neighb_addr_mask)
                # i_ct_normalized = normalize(i_ct) * 255
                # i_mask_normalized = (i_mask * 255).astype(np.uint8)
                # i_gt_overlay = overlay(gray2rgb(i_ct_normalized), i_mask_normalized, (0, 255, 0), 0.3)
                # plt.imshow(i_gt_overlay)
                # plt.show()

                cts.append(i_ct / 3071)
                suvs.append(np.load(i_neighb_addr_suv))
                masks.append(i_mask)
        else:
            search_idx = self.Index[patient]
            for i in range(self.num_nodes - 1):
                if idx_base + 10 < search_idx[1] and idx_base - 10 >= search_idx[0]:
                    idx = np.random.randint(idx_base - 10, idx_base + 10)
                elif idx_base + 10 >= search_idx[1] and idx_base - 10 >= search_idx[0]:
                    loc = search_idx[1] - idx_base
                    idx = np.random.randint(idx_base - (10 + (10 - loc)), idx_base + loc)
                elif idx_base + 10 < search_idx[1] and idx_base - 10 < search_idx[0]:
                    loc = idx_base - search_idx[0]
                    idx = np.random.randint(idx_base - loc, idx_base + (10 + (10 - loc)))

                file_path = self.all_files[idx]
                mask_path = self.change_img_to_label_path(file_path)
                suv_path = self.change_img_to_suv_path(file_path)
                ct = (np.load(file_path)) / 3071
                cts.append(ct)
                masks.append(np.load(mask_path))
                suvs.append(np.load(suv_path))

        if self.augment_params:
            cts, suvs, masks = self.augment(cts, suvs, masks)

        nodes = []
        for i in range(len(cts)):
            concat_img = np.zeros([2, 256, 256])
            concat_img[0, :, :] = suvs[i]
            concat_img[1, :, :] = cts[i]
            concat_img = concat_img.astype('float32')
            nodes.append(concat_img)
            masks[i] = np.expand_dims(masks[i], 0).astype('float32')

        return nodes, masks
