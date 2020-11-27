import glob
import cv2 
import numpy as np

# train human image resize and save
path1 = glob.glob("C:/Users/bitcamp/Desktop/intel image classification/train/human_data/*.jpg")
cv_img1 = []
index1 = 0
for img in path1:
    read_image1 = cv2.imread(img)
    resize_image1 = cv2.resize(read_image1, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    bgr_to_rgb1 = cv2.cvtColor(resize_image1, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/bitcamp/Desktop/intel image classification/train/human/'+ str(index1) +'.jpg' , bgr_to_rgb1)
    cv_img1.append(resize_image1)
    index1 += 1
print("train 불러오기, resize 완료!")

train_images = np.array(cv_img1)
print(train_images.shape)            # (2000, 150, 150, 3)

# test human image resize and save
path2 = glob.glob("C:/Users/bitcamp/Desktop/intel image classification/test/human_data/*.jpg")
cv_img2 = []
index2 = 0
for img in path2:
    read_image2 = cv2.imread(img)
    resize_image2 = cv2.resize(read_image2, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    bgr_to_rgb2 = cv2.cvtColor(resize_image2, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/bitcamp/Desktop/intel image classification/test/human/'+ str(index2) +'.jpg', bgr_to_rgb2)
    cv_img2.append(resize_image2)
    index2 += 1
print("test 불러오기, resize 완료!")

test_images = np.array(cv_img2)
print(test_images.shape)             # (400, 150, 150, 3)

# predict human image resize and save
path3 = glob.glob("C:/Users/bitcamp/Desktop/predict/*.jpg")
cv_img3 = []
index3 = 25000
for img in path3:
    read_image3 = cv2.imread(img)
    resize_image3 = cv2.resize(read_image3, dsize=(150, 150), interpolation=cv2.INTER_AREA)
    bgr_to_rgb3 = cv2.cvtColor(resize_image3, cv2.COLOR_BGR2RGB)
    cv2.imwrite('C:/Users/bitcamp/Desktop/intel image classification/pred/'+ str(index3) +'.jpg', bgr_to_rgb3)
    cv_img3.append(resize_image3)
    index3 += 1

print("predict 불러오기, resize 완료!")

predict_images = np.array(cv_img3)
print(predict_images.shape)        # (1000, 150, 150, 3)     