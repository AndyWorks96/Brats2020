import FcnTest
import UnetTest
import ThreeDUnet_test
import ThreeDVNet_test
from GetTesting import testSet
from ThreeDProprecessing import test3DSet


class Segmetation:

    def __init__(self, dir_path):
        
        self.dir_path = dir_path

    def FcnPre(self):

        gettest = testSet(self.dir_path)
        gettest.sub()
        fcn=FcnTest.main()
        return fcn

    def UnetPre(self):

        gettest = testSet(self.dir_path)
        gettest.sub()
        unet = UnetTest.main()
        return unet

    def Unet3dPre(self):

        gettest = test3DSet(self.dir_path)
        gettest.sub()
        unet3d = ThreeDUnet_test.main()
        return unet3d

    def Vnet3dPre(self):

        gettest = test3DSet(self.dir_path)
        gettest.sub()
        vnet3d = ThreeDVNet_test.main()
        return vnet3d


if __name__ == '__main__':
    seg=Segmetation('./data/HGG')
    # seg.FcnPre()
    # seg.UnetPre()
    # seg.Unet3dPre()
    seg.Vnet3dPre()
    print(seg.dir_path)
