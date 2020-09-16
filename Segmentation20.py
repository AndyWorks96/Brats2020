import FcnTest
import UnetTest
from GetTesting import testSet

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


if __name__ == '__main__':
    seg=Segmetation('./data/HGG')
    seg.FcnPre()
    # seg.UnetPre()
    print(seg.dir_path)
