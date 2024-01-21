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
from unet_predictor import UNet
import imutils
from scipy.spatial import distance as dist


def normalize(img):
    min = np.min(img)
    max = np.max(img)
    img = img - (min)
    img = img / (max - min)
    return img


class LungDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment_params, num_nodes=5, eval=False):
        self.all_files = self.extract_files(root)
        self.augment_params = augment_params
        self.num_nodes = num_nodes
        weights = \
        torch.load("/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/ct_suv/logs/epoch_40.pth", map_location='cuda')[
            'model']
        unet_model = UNet().cuda()
        unet_model.load_state_dict(weights)
        self.unet = unet_model

        self.center_contours = pickle.load(
            open(
                '/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/min_distance_neighbouring/patient_train_center',
                'rb'))

    def predict_mask(self, data):
        with torch.no_grad():
            inpt = torch.tensor(data).cuda().unsqueeze(0)
            output = self.unet(inpt)
            output = torch.sigmoid(output.squeeze())
            output = output > 0.5
            output = output.cpu().numpy()
        return output

    def min_neighbours(self, target_mask, num_neighbours):
        # target_mask = target_mask.astype('uint')
        target_mask = target_mask.astype(np.uint8)
        ref_contours = cv2.findContours(target_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_contours = imutils.grab_contours(ref_contours)
        other_mask_centers = self.center_contours.copy()

        neighbours = []
        for node in range(num_neighbours):
            cal_dist = []
            for cont in ref_contours:
                (center_x, center_y), (width, height), angle = cv2.minAreaRect(cont)
                min_dist = 10000
                for indx, other_mask in enumerate(other_mask_centers):
                    search_center_x = other_mask['center_x']
                    search_center_y = other_mask['center_y']
                    distance = dist.euclidean((center_x, center_y), (search_center_x, search_center_y))
                    if distance < min_dist:
                        min_dist = distance
                        search_p = other_mask
                        indx_search_p = indx
                if min_dist != 10000:
                    cal_dist.append({'p_info': search_p, 'dis': min_dist})
                    del other_mask_centers[indx_search_p]

            arr_dist = []
            for item in cal_dist:
                arr_dist.append(item['dis'])
            neighbours.append(cal_dist[np.argmin(arr_dist)])

        return neighbours

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
        # file_path = Path('/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/val/3130125/ct/100.npy')
        patient = file_path.parts[-3]
        mask_name = file_path.parts[-1]

        target_mask_path = self.change_img_to_label_path(file_path)
        target_suv_path = self.change_img_to_suv_path(file_path)
        target_ct = (np.load(file_path)) / 3071
        target_mask = np.load(target_mask_path)
        cts.append(target_ct)
        masks.append(target_mask)
        suvs.append(np.load(target_suv_path))

        concat_img = np.zeros([2, 256, 256])
        concat_img[0, :, :] = suvs[0]
        concat_img[1, :, :] = cts[0]
        concat_img = concat_img.astype('float32')
        target_pred_mask = self.predict_mask(concat_img)
        if np.any(target_pred_mask):
            neighbours = self.min_neighbours(target_pred_mask, 2)
            for i_neighb in neighbours:
                i_neighb_addr_ct = os.path.join(root, i_neighb['p_info']['p_name'], 'ct',
                                                i_neighb['p_info']['mask_name'])
                i_neighb_addr_suv = os.path.join(root, i_neighb['p_info']['p_name'], 'suv',
                                                 i_neighb['p_info']['mask_name'])
                i_neighb_addr_mask = os.path.join(root, i_neighb['p_info']['p_name'], 'mask',
                                                  i_neighb['p_info']['mask_name'])
                cts.append(np.load(i_neighb_addr_ct) / 3071)
                suvs.append(np.load(i_neighb_addr_suv))
                masks.append(np.load(i_neighb_addr_mask))
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
        for i in range(self.num_nodes):
            concat_img = np.zeros([2, 256, 256])
            concat_img[0, :, :] = suvs[i]
            concat_img[1, :, :] = cts[i]
            concat_img = concat_img.astype('float32')
            nodes.append(concat_img)
            masks[i] = np.expand_dims(masks[i], 0).astype('float32')

        return nodes, masks
