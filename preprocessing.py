import os
import numpy as np
import cv2
import imutils
import pickle

root = '/media/fumcomp/9CAA7029AA6FFDD8/fold_0/project/data/train/'
patients = os.listdir(root)
mask_center = []
for p in patients:
    mask_p_names = os.listdir(os.path.join(root, p, 'mask'))
    for mask_name in mask_p_names:
        mask = np.load(os.path.join(root, p, 'mask', mask_name)).astype(np.uint8)
        if np.any(mask):
            ref_contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ref_contours = imutils.grab_contours(ref_contours)
            for cont in ref_contours:
                (center_x, center_y), (width, height), angle = cv2.minAreaRect(cont)
                mask_center.append({'p_name': p, 'mask_name': mask_name, 'center_x': center_x, 'center_y': center_y})
pickle.dump(mask_center, open('patient_train_center', 'wb'))
