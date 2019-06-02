from mxnet.gluon import nn, HybridBlock
import numpy as np
import random
import math

class CutOut(nn.Block):
    """ Randomly mask out one or more patches from an image.
        Args:
            n_holes(int): Number of patches to cut out of each image
            length (int): The length (in pixels) of each square patches
    """
    def __init__(self, length, n_holes=1):
        print('Use cutout...')
        super(CutOut, self).__init__()
        self.length=length
        self.n_holes=n_holes

    def forward(self, img):
        for n in range(self.n_holes):
            y = np.random.randint(0, img.shape[0]-1)
            x = np.random.randint(0, img.shape[1]-1)
            y1 = np.clip(y - self.length//2, 0, img.shape[0])
            y2 = np.clip(y + self.length//2, 0, img.shape[0])
            x1 = np.clip(x - self.length//2, 0, img.shape[1])
            x2 = np.clip(x + self.length//2, 0, img.shape[1])
            img[y1:y2, x1:x2] = 0
        return img



class RandomErasing(nn.Block):
    """Random erasing the an rectangle region in Image
    
    Args:
        p :   The probability that the operation will be performed.
        s1:   min erasing area
        sh:   max erasing area
        r1:   min aspect ratio range of earsing region
    """
    def __init__(self, p=0.5, s1=0.02, sh=0.4, r1=0.3):
        print('Use RandomErasing...')
        super(RandomErasing, self).__init__()
        self.p = p
        self.s = (s1,sh)
        self.r = (r1, 1/r1)
        
    def forward(self, img):
        """
        Args:
            img: img with shape (h,w,c)
        """
        assert len(img.shape)==3, 'image should be 3-dimensional array'
        if random.random() < self.p:
            return img
        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r) 

                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))

                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])

                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye : ye + He, xe : xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))

                    return img
        
        
        
        
