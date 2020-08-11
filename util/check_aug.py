import cv2
import os
import util.augmentations as aug
import math
import numpy

image_path = "E:\Code\PSPnet\image\BF1.jpg"
label_path = image_path

def main(img, label):
    cv2.imshow("1", img)
    func1 = aug.Crop(100, 'random')
    func2 = aug.RandomGaussianBlur()
    func3 = aug.RandomVerticalFlip()
    func4 = aug.RandomHorizontalFlip()
    funcs = aug.RandomResize(size=[0.5, 2])
    funcr = aug.RandomRotate(rotate=[-10, 10], padding=[255,255,255])
    resize = aug.Resize((224,224))
    norm = aug.Normalize([128])
    tt = aug.ToTensor()
    tn = aug.ToNumpy()
    op = aug.Compose([tt, norm, tn])
    t1 = aug.BGR2RGB()
    t2 = aug.RGB2BGR()

    img1, label1 = func1(img, label)
    img2, label2 = func2(img, label)
    img3, label3 = func3(img, label)
    img4, label4 = func4(img, label)
    imgs, labels = funcs(img, label)
    imgr, labelr = funcr(img, label)
    img5, label5 = resize(img, label)
    imgn, labeln = op(img, label)
    img6, label6 = t1(img, label)
    img7, label7 = t2(img6, label6)


    cv2.imshow("aug", img1)
    cv2.imshow("label", label1)
    #cv2.imshow("t1", img6)
    #cv2.imshow("t2", img7)
    cv2.waitKey(0)



if __name__ == '__main__':
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    print(os.getcwd())
    main(img, label)
