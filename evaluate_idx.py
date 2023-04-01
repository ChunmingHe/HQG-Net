import cv2
import numpy as np


def SNRLossnp(y_true, y_pred):
    y_mean = np.mean(y_true)
    y_var = np.var(y_true)

    y_std = np.sqrt(y_var)

    print(y_mean, y_var, y_std)

    return 10*np.log10(np.max(y_pred) ** 2 / y_std ** 2)


img_path1 = "data/CORN_2/testA/batch_1_1.tif"
img_path2 = "evaluation/co_result/visualization/aug_batch_1_1.png"
mask_origin_path = "./data/CORN_2/testB/batch_1_1.tif"
mask_ground_path = "./mask_ground_3/batch_1_1.jpg"

threshold = 127

img = cv2.imread(img_path2, 0)
img = cv2.resize(img, (384, 384))
mask_origin = cv2.imread(mask_origin_path, 0)
mask_ground = cv2.imread(mask_ground_path, 0)
# print(mask_origin)
# print(np.nonzero(mask_origin))
[x, y] = np.nonzero(mask_origin)
# print(x,y)


max_list = []
sigma_list = []

mask_back = img.copy()
mask_target = np.zeros([384, 384])


signal_list = []
back_list = []
for i in range(len(x)):
    mask_target[x[i], y[i]] = img[x[i], y[i]]


for i in range(384):
    for j in range(384):
        if mask_ground[i, j] < threshold:
            mask_back[i, j] = 0
print(mask_target.shape)
print(mask_back.shape)
SNR = SNRLossnp(mask_back, mask_target)

print(SNR)

# cv2.imshow('mask_back', mask_back)
#
# cv2.waitKey(5000)
#
# cv2.imshow('mask_target', mask_target)
#
# cv2.waitKey(5000)
#
# cv2.imshow('img',img)
#
# cv2.waitKey(5000)
# cv2.imshow('image',mask_origin)

# cv2.waitKey(1000) 

# print(mask_ground.shape)

# #img = img.transpose([2, 0, 1])
# #mask_origin = mask_origin.transpose([2, 0, 1])
# #mask_ground = mask_ground.transpose([2, 0, 1])

# max_list = []
# sigma_list = []

# # for i in zip(range(img.shape[0])):
# #     signal_list = []
# #     noise_list = []
# #     for x in range(img.shape[1]):
# #         # for y in range(img.shape[2]):
# #         #     if mask_origin[i, x, y] > threshold:
# #         #         signal_list.append(img[i, x, y])
# #         #     if mask_ground[i, x, y] > threshold:
# #         #         noise_list.append(img[i, x, y])
# #         for y in range(img.shape[2]):
# #             if mask_origin[i, x, y] > threshold:
# #                 signal_list.append(img[i, x, y])
# #             if mask_ground[i, x, y] > threshold:
# #                 noise_list.append(img[i, x, y])







# signal_list = []
# noise_list = []
# for x in range(img.shape[0]):
#     # for y in range(img.shape[2]):
#     #     if mask_origin[i, x, y] > threshold:
#     #         signal_list.append(img[i, x, y])
#     #     if mask_ground[i, x, y] > threshold:
#     #         noise_list.append(img[i, x, y])
#     for y in range(img.shape[1]):
#         if mask_origin[ x, y] > threshold:
#             signal_list.append(img[ x, y])
#         if mask_ground[ x, y] > threshold:
#             noise_list.append(img[x, y])



# signal_max = np.max(signal_list)
# noise_sigma = np.std(back_list)

# max_list.append(signal_max)
# sigma_list.append(noise_sigma)

# Is = np.max(max_list)
# SigmaB = np.mean(sigma_list)

# print("最大亮度：{}， 背景标准差：{}".format(Is, SigmaB))
# SNR = 10 * np.log10(Is ** 2 / SigmaB ** 2)
# print("SNR:{}".format(SNR))
