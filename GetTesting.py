import os
import numpy as np
import SimpleITK as sitk


class testSet:

    out_path='./data/HGG'
    def __init__(self, out_path):
        self.dir_path = out_path

        self.flair_name = "_flair.dcm"
        self.t1_name = "_t1.dcm"
        self.t1ce_name = "_t1ce.dcm"
        self.t2_name = "_t2.dcm"
        self.mask_name = "_seg.dcm"

    # bratslgg_path = './data/LGG'




    # 此处加上self报错：self和file_dir
    def file_name_path(self,file_dir, dir=True, file=False):
        """
        get root path,sub_dirs,all_sub_files
        :param file_dir:
        :return: dir or file
        """
        for root, dirs, files in os.walk(file_dir):
            if len(dirs) and dir:
                print("sub_dirs:", dirs)
                return dirs
            if len(files) and file:
                print("files:", files)
                return files

    # pathhgg_list = file_name_path(self.dir_path)


    def normalize(self,slice, bottom=99, down=1):
        """
        normalize image with mean and std for regionnonzero,and clip the value into range
        :param slice:
        :param bottom:
        :param down:
        :return:
        """
        #有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
        b = np.percentile(slice, bottom)
        t = np.percentile(slice, down)
        slice = np.clip(slice, t, b)#限定范围numpy.clip(a, a_min, a_max, out=None)

        #除了黑色背景外的区域要进行标准化
        image_nonzero = slice[np.nonzero(slice)]
        if np.std(slice) == 0 or np.std(image_nonzero) == 0:
            return slice
        else:
            tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            # since the range of intensities is between 0 and 5000 ,
            # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            # the min is replaced with -9 just to keep track of 0 intensities
            # so that we can discard those intensities afterwards when sampling random patches
            tmp[tmp == tmp.min()] = -9 #黑色背景区域
            return tmp

    def crop_ceter(self,img,croph,cropw):
        #for n_slice in range(img.shape[0]):
        height,width = img[0].shape
        starth = height//2-(croph//2)
        startw = width//2-(cropw//2)
        return img[:,starth:starth+croph,startw:startw+cropw]

    def sub(self):

        pathhgg_list = self.file_name_path(self.dir_path)
        outputImg_path = './data/testImageDcm'
        outputMask_path = './data/testMaskDcm'
        if not os.path.exists(outputImg_path):
            os.mkdir(outputImg_path)
        if not os.path.exists(outputMask_path):
            os.mkdir(outputMask_path)

        for subsetindex in range(len(pathhgg_list)):
            brats_subset_path = self.dir_path + "/" + str(pathhgg_list[subsetindex]) + "/"
            # 获取每个病例的四个模态及Mask的路径
            flair_image = brats_subset_path + str(pathhgg_list[subsetindex]) + self.flair_name
            t1_image = brats_subset_path + str(pathhgg_list[subsetindex]) + self.t1_name
            t1ce_image = brats_subset_path + str(pathhgg_list[subsetindex]) + self.t1ce_name
            t2_image = brats_subset_path + str(pathhgg_list[subsetindex]) + self.t2_name
            mask_image = brats_subset_path + str(pathhgg_list[subsetindex]) + self.mask_name
            # 获取每个病例的四个模态及Mask数据
            flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
            t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
            t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
            t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
            mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
            # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
            flair_array = sitk.GetArrayFromImage(flair_src)
            t1_array = sitk.GetArrayFromImage(t1_src)
            t1ce_array = sitk.GetArrayFromImage(t1ce_src)
            t2_array = sitk.GetArrayFromImage(t2_src)
            mask_array = sitk.GetArrayFromImage(mask)
            # 对四个模态分别进行标准化,由于它们对比度不同
            flair_array_nor = self.normalize(flair_array)

            t1_array_nor = self.normalize(t1_array)
            t1ce_array_nor = self.normalize(t1ce_array)
            t2_array_nor = self.normalize(t2_array)
            # 裁剪(偶数才行)
            flair_crop = self.crop_ceter(flair_array_nor, 160, 160)
            t1_crop = self.crop_ceter(t1_array_nor, 160, 160)
            t1ce_crop = self.crop_ceter(t1ce_array_nor, 160, 160)
            t2_crop = self.crop_ceter(t2_array_nor, 160, 160)
            mask_crop = self.crop_ceter(mask_array, 160, 160)
            flair_crop = flair_crop
            t1_crop = t1_crop
            t1ce_crop = t1ce_crop
            t2_crop = t2_crop
            mask_crop = mask_crop

            print(str(pathhgg_list[subsetindex]))
            # 切片处理,并GetTestingSets.py
            for n_slice in range(flair_crop.shape[0]):
                if np.max(mask_crop[n_slice, :, :]) != 0:
                    # if np.max(mask_crop[n_slice, :, :]) ==4:
                        # print("*******")
                        # print(mask_crop[n_slice,::][110])
                        # print(np.max(mask_crop[n_slice,:,:]))
                        # print("*******")
                    maskImg = mask_crop[n_slice, :, :]

                    FourModelImageArray = np.zeros((flair_crop.shape[1], flair_crop.shape[2], 4), np.float)
                    flairImg = flair_crop[n_slice, :, :]
                    flairImg = flairImg.astype(np.float)
                    FourModelImageArray[:, :, 0] = flairImg
                    t1Img = t1_crop[n_slice, :, :]
                    t1Img = t1Img.astype(np.float)
                    FourModelImageArray[:, :, 1] = t1Img
                    t1ceImg = t1ce_crop[n_slice, :, :]
                    t1ceImg = t1ceImg.astype(np.float)
                    FourModelImageArray[:, :, 2] = t1ceImg
                    t2Img = t2_crop[n_slice, :, :]
                    t2Img = t2Img.astype(np.float)
                    FourModelImageArray[:, :, 3] = t2Img

                    imagepath = outputImg_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    maskpath = outputMask_path + "\\" + str(pathhgg_list[subsetindex]) + "_" + str(n_slice) + ".npy"
                    np.save(imagepath, FourModelImageArray)  # (160,160,4) np.float dtype('float64')
                    np.save(maskpath, maskImg)  # (160, 160) dtype('uint8') 值为0 1 2 4

if __name__ == '__main__':

    out_path = './data/HGG'
    gettest=testSet(out_path)
    gettest.sub()
