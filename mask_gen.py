import numpy as np
import cv2
import os


IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
    'tif', 'TIF', 'tiff', 'TIFF',
]


def is_image_file(filename):
    return True if filename.split('.')[-1] in IMG_EXTENSIONS else False


def load_images(root):
    images = []
    for filename in os.listdir(root):
        if is_image_file(filename):
            path = os.path.join(root, filename)
            images.append(path)
    return images


def generate_mask(file_root, dilation_radius):

    img_list = load_images(file_root)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius, dilation_radius))

    save_path = "./mask_ground_{}".format(dilation_radius)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for f_pth in img_list:
        img = cv2.imread(f_pth)
        mask_dilation = cv2.dilate(img, kernel)
        mask_background = mask_dilation - img
        # mask_background = mask_dilation
        img_save_path = os.path.join(save_path, (f_pth.split("\\")[-1]).split('.')[0])
        img_save_path = img_save_path + ".jpg"
        cv2.imwrite(img_save_path, mask_background)


if __name__ == "__main__":
    dilation_r = 3
    img_root = "data/CORN_2/testB"
    generate_mask(img_root, dilation_r)
