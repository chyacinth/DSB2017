import numpy as np
from PIL import Image
import os
from multiprocessing import Pool
import SimpleITK as sitk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
#
# def matrix2int16(matrix):
#     '''
# matrix must be a numpy array NXN
# Returns uint16 version
#     '''
#     m_min= np.min(matrix)
#     m_max= np.max(matrix)
#     matrix = matrix-m_min
#     return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))
#
#
# # Load npy
# # img_array_npy = np.load('/Users/hyacinth/PycharmProjects/DSB2017-master/prep_result/0b922b907eeb8f02010c876f0c2efe26_clean.npy')
# # img_array = np.load('/Users/hyacinth/PycharmProjects/DSB2017-master/prep_result/0b922b907eeb8f02010c876f0c2efe26_label.npy')
# # print(img_array_npy.shape)
# # img_bitmap = Image.fromarray(img_array_npy[0][100])
# # Image._show(img_bitmap)
#
# # Load .mhd
# file_name = '/Users/hyacinth/Downloads/tianchi/val_subset00/LKDS-00121.mhd'
# img = sitk.ReadImage(file_name)
# img_array = sitk.GetArrayFromImage(img)
# img_bitmap = Image.fromarray(img_array[0])
# Image._show(img_bitmap)
# print(img_array.shape)
# file_name = '/Volumes/Hyacinth/Dropbox/Dropbox/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.621916089407825046337959219998.mhd'
# img = sitk.ReadImage(file_name)
# img_array = sitk.GetArrayFromImage(img)
# print(img_array.shape)
# imgs = np.ndarray([3, 512, 512], dtype=np.float32)
# imgs[0] = matrix2int16(img_array[80])
# imgs[0] = img_array[80]
# print(img_array[80][0][0])
# print(imgs[0][0][0])
# fig,ax = plt.subplots(2,2,figsize=[8,8])
#
#
# img = imgs[0]
# #Standardize the pixel values
# mean = np.mean(img)
# std = np.std(img)
# img = img-mean
# img = img/std
# plt.hist(img.flatten(),bins=200)
#
# middle = img[100:400,100:400]
# mean = np.mean(middle)
# max = np.max(img)
# min = np.min(img)
# # move the underflow bins
# img[img==max]=mean
# img[img==min]=mean
# kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
# centers = sorted(kmeans.cluster_centers_.flatten())
# threshold = np.mean(centers)
# thresh_img = np.where(img < threshold,1.0,0.0)  # threshold the image
#
# # Erosion and Dilation
# eroded = morphology.erosion(thresh_img,np.ones([4,4]))
# dilation = morphology.dilation(eroded,np.ones([10,10]))
# labels = measure.label(dilation)
# label_vals = np.unique(labels)
# plt.imshow(labels)
#
# labels = measure.label(dilation)
# label_vals = np.unique(labels)
# regions = measure.regionprops(labels)
# good_labels = []
# for prop in regions:
#     B = prop.bbox
#     if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
#         good_labels.append(prop.label)
# mask = np.ndarray([512,512],dtype=np.int8)
# mask[:] = 0
# #
# #  The mask here is the mask for the lungs--not the nodes
# #  After just the lungs are left, we do another large dilation
# #  in order to fill in and out the lung mask
# #
# for N in good_labels:
#     mask = mask + np.where(labels==N,1,0)
# mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
# plt.imshow(mask,cmap='gray')
#
# #masks = np.load(working_path+"lungmask_0.py")
# #imgs = np.load(working_path+"images_0.py")
# img = mask*img
#
# #
# # renormalizing the masked image (in the mask region)
# #
# new_mean = np.mean(img[mask>0])
# new_std = np.std(img[mask>0])
# #
# #  Pushing the background color up to the lower end
# #  of the pixel range for the lungs
# #
# old_min = np.min(img)       # background color
# img[img==old_min] = new_mean-1.2*new_std   # resetting backgound color
# img = img-new_mean
# img = img/new_std
#
# ax[0,0].imshow(img,cmap='gray')
# ax[0,1].imshow(img_array_npy[0][100],cmap='gray')
# plt.show()

from preprocessing.full_prep import savenpy_mhd
filelist = []
filelist += [each for each in os.listdir('/Volumes/Hyacinth/tianchi/val/val_subset00') if each.endswith('.mhd')]
pool = Pool(None)

savenpy_mhd(1,filelist,'/Users/hyacinth/Downloads/tianchi/test/prep_data','/Volumes/Hyacinth/tianchi/val/val_subset00')

img_array_npy = np.load('/Users/hyacinth/Downloads/tianchi/test/prep_data/'+filelist[0]+'_clean.npy')
# img_array_npy = np.load('/Users/hyacinth/PycharmProjects/DSB2017-master/prep_result/0b8afe447b5f1a2c405f41cf2fb1198e_clean.npy')
print(img_array_npy.shape)
img_bitmap = Image.fromarray(img_array_npy[0][150])
Image._show(img_bitmap)