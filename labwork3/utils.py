import numpy as np
from numba import cuda

def grayscaleCPU(image):
    for i in range(image.shape[0]):
        gray = np.uint8((int(image[i, 0]) + int(image[i, 1]) + int(image[i, 2])) / 3)
        image[i, 0] = gray
        image[i, 1] = gray
        image[i, 2] = gray
    return image

@cuda.jit
def grayscaleGPU(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

