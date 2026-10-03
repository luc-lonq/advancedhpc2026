import math

from numba import cuda
import numba
from matplotlib import pyplot as plt
import numpy as np

@cuda.jit
def histogram(src, hist):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx >= src.shape[0]:
        return

    g = int(src[tidx])

    cuda.atomic.add(hist, g, 1)



img = plt.imread('img/image.jpg')
flat_img = img.reshape(-1)
devInput = cuda.to_device(flat_img)
n = img.shape[0] * img.shape[1]
devHist = cuda.to_device(np.zeros(256, dtype=np.int32))
blockSize = 1024
gridSize = math.ceil(n / blockSize)
histogram[gridSize, blockSize](devInput, devHist)
hist = devHist.copy_to_host()
plt.bar(range(256), hist)
plt.title('Histogram of Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('img/histogram.png')
plt.show()

@cuda.jit
def cdf_kernel(hist, cdf):
    shared = cuda.shared.array(256, dtype=numba.int32)
    tid = cuda.threadIdx.x
    shared[tid] = hist[tid]
    cuda.syncthreads()

    offset = 1

    while offset < 256:
        if tid >= offset:
            value = shared[tid - offset]
        else:
            value = 0
        cuda.syncthreads()
        if tid >= offset:
            shared[tid] += value
        cuda.syncthreads()

        offset *= 2

    cdf[tid] = shared[tid]

devInput = cuda.to_device(hist)
devCdf = cuda.device_array(256, dtype=np.int32)
blockSize = 256
gridSize = math.ceil(256 / blockSize)
cdf_kernel[gridSize, blockSize](devInput, devCdf)
cdf_result = devCdf.copy_to_host()
plt.bar(range(256), cdf_result)
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Pixel Value')
plt.ylabel('Cumulative Frequency')
plt.savefig('img/cdf.png')
plt.show()

cdf_min = cdf_result[cdf_result > 0][0]
print(f"Minimum non-zero CDF value: {cdf_min}")


@cuda.jit
def lut_kernel(cdf, cdf_min, n, lut):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx >= 256:
        return

    lut[tidx] = max(0, round((cdf[tidx] - cdf_min) / (n - cdf_min) * 255))


devInput = cuda.to_device(cdf_result)
devLut = cuda.device_array(256, dtype=np.uint8)
blockSize = 256
gridSize = math.ceil(256 / blockSize)
lut_kernel[gridSize, blockSize](devInput, cdf_min, n, devLut)
lut = devLut.copy_to_host()
print(lut)

@cuda.jit
def apply_lut(src, dst, lut):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx >= src.shape[0]:
        return

    g = int(src[tidx])
    dst[tidx] = lut[g]

devInput = cuda.to_device(flat_img)
devOutput = cuda.device_array((n,), dtype=np.uint8)
blockSize = 1024
gridSize = math.ceil(n / blockSize)
apply_lut[gridSize, blockSize](devInput, devOutput, devLut)
equalized_img = devOutput.copy_to_host().reshape(img.shape)
plt.imshow(equalized_img, cmap='gray')
plt.show()
plt.imsave('img/image_equalized.png', equalized_img, cmap='gray')