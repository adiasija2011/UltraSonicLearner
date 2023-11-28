import os
import cv2

img_dir = './data/busi/images/'
mask_dir = './data/busi/masks/0/'

def find_extra_mask(im_name):
    l = []
    for mask in os.listdir(mask_dir):
        sp_name = mask.split('_')
        if im_name == sp_name[0] + '.png':
            if len(sp_name) == 2:
                l.append([mask, cv2.imread(mask_dir + mask)])
    return l

w_dir = './data/busi/combined_masks/0/'

if not os.path.exists(w_dir):
    os.makedirs(w_dir)

for img in os.listdir(img_dir):
    first_mask_name = mask_dir + img
    l = find_extra_mask(img)
    first_mask = cv2.imread(first_mask_name)
    new_mask = first_mask.copy()
    if l:
        for pair in l:
            additional_mask_name, additional_mask = pair
            new_mask = cv2.add(new_mask, additional_mask)
        cv2.imwrite(w_dir + img, new_mask)
