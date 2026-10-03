import math

from numba import cuda
import numba
import numpy as np
from matplotlib import pyplot as plt

@cuda.jit
def rgb_to_hsv(src, dstH, dstS, dstV):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx >= src.shape[0]:
        return

    r = src[tidx, 0] / 255.0
    g = src[tidx, 1] / 255.0
    b = src[tidx, 2] / 255.0

    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        h = 0.0
    elif cmax == r:
        h = 60 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60 * (((b - r) / delta) + 2)
    else:
        h = 60 * (((r - g) / delta) + 4)

    s = 0.0 if cmax == 0 else (delta / cmax)

    v = cmax

    dstH[tidx] = h
    dstS[tidx] = s
    dstV[tidx] = v



img = plt.imread('img/image.jpg')
flat_img = img.reshape((img.shape[0] * img.shape[1], 3))
# AoS
devInput = cuda.to_device(flat_img)
# SoA
n = img.shape[0] * img.shape[1]
devH = cuda.device_array(n, dtype=np.float32)
devS = cuda.device_array(n, dtype=np.float32)
devV = cuda.device_array(n, dtype=np.float32)
blockSize = 1024
gridSize = math.ceil(n / blockSize)

rgb_to_hsv[gridSize, blockSize](devInput, devH, devS, devV)

hsv_img = {
    'H': devH.copy_to_host(),
    'S': devS.copy_to_host(),
    'V': devV.copy_to_host()
}

print(hsv_img)


@cuda.jit
def hsv_to_rgb(srcH, srcS, srcV, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if tidx >= srcH.shape[0]:
        return
    r, g, b = 0, 0, 0


    h = srcH[tidx]
    s = srcS[tidx]
    v = srcV[tidx]

    d = h / 60
    hi = int(d) % 6
    f = d - int(d)
    l = v * (1 - s)
    m = v * (1 - f * s)
    n = v * (1 - (1 - f) * s)

    if h >= 0 and h < 60:
        r, g, b = v, n, l
    elif h >= 60 and h < 120:
        r, g, b = m, v, l
    elif h >= 120 and h < 180:
        r, g, b = l, v, n
    elif h >= 180 and h < 240:
        r, g, b = l, m, v
    elif h >= 240 and h < 300:
        r, g, b = n, l, v
    elif h >= 300 and h < 360:
        r, g, b = v, l, m

    dst[tidx, 0] = int(r * 255)
    dst[tidx, 1] = int(g * 255)
    dst[tidx, 2] = int(b * 255)

devH = cuda.to_device(hsv_img['H'])
devS = cuda.to_device(hsv_img['S'])
devV = cuda.to_device(hsv_img['V'])
devOutput = cuda.device_array(flat_img.shape, dtype=np.uint8)
hsv_to_rgb[gridSize, blockSize](devH, devS, devV, devOutput)
rgb_img = devOutput.copy_to_host().reshape(img.shape)
plt.imshow(rgb_img)
plt.imsave("img/image_rgb_after_conversion.jpg", rgb_img)