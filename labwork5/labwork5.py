import math
import time
import numba
import numpy as np
from numba import cuda
from matplotlib import pyplot as plt

GAUSSIAN_BLUR_MATRIX = [
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
]

GAUSSIAN_BLUR_MATRIX_SUM = 1003

def gaussian_blur_cpu(image):
    h, w, chan = image.shape
    output_image = np.zeros_like(image)
    image = image.astype(np.int32)

    for y in range(h):
        for x in range(w):
            for c in range(chan):
                out_of_range_pixel_value = 0
                pixel_value = 0
                for ky in range(-3, 4):
                    for kx in range(-3, 4):
                        ly = y + ky
                        lx = x + kx
                        if ly < 0 or ly >= h or lx < 0 or lx >= w:
                            out_of_range_pixel_value += GAUSSIAN_BLUR_MATRIX[ky + 3][kx + 3]
                        else:
                            pixel_value += image[ly, lx, c] * GAUSSIAN_BLUR_MATRIX[ky + 3][kx + 3]
                output_image[y, x, c] = pixel_value // (GAUSSIAN_BLUR_MATRIX_SUM - out_of_range_pixel_value)

    return output_image

img = plt.imread("img/thuya.jpg")
if img.dtype != np.uint8:  # PNG files are loaded as float in [0, 1]
    img = (img * 255).astype(np.uint8)
t1 = time.time()
blurred_img = gaussian_blur_cpu(img)
t2 = time.time()
print(f"CPU time: {t2 - t1:.2f} seconds")
plt.imshow(blurred_img)
plt.imsave("img/img_blurred_cpu.jpg", blurred_img)

@cuda.jit
def gaussian_blur_gpu(src, dst, matrix, matrix_sum):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    h, w, chan = src.shape
    if tidx >= w or tidy >= h:
        return
    for c in range(chan):
        out_of_range_pixel_value = 0
        pixel_value = 0
        for ky in range(-3, 4):
            for kx in range(-3, 4):
                ly = tidy + ky
                lx = tidx + kx
                if ly < 0 or ly >= h or lx < 0 or lx >= w:
                    out_of_range_pixel_value += matrix[ky + 3, kx + 3]
                else:
                    pixel_value += src[ly, lx, c] * matrix[ky + 3, kx + 3]
        dst[tidy, tidx, c] = pixel_value // (matrix_sum - out_of_range_pixel_value)

img = plt.imread("img/img.jpg")
hostInput = np.zeros(img.shape, np.uint8)
devInput = cuda.to_device(img)
hostOutput = np.zeros(img.shape, np.uint8)
devOutput = cuda.to_device(hostOutput)
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))
devMatrix = cuda.to_device(np.array(GAUSSIAN_BLUR_MATRIX, dtype=np.int32))
t1 = time.time()
gaussian_blur_gpu[gridSize, blockSize](devInput, devOutput, devMatrix, GAUSSIAN_BLUR_MATRIX_SUM)
cuda.synchronize()
t2 = time.time()
hostOutput = devOutput.copy_to_host()
plt.imshow(hostOutput)
plt.imsave("img/img_blurred_gpu.jpg", hostOutput)
print(f"GPU time: {t2 - t1:.2f} seconds")


@cuda.jit
def gaussian_blur_gpu_shared_memory(src, dst, matrix, matrix_sum):
    tile = cuda.shared.array(shape=(32, 32, 3), dtype=numba.uint8)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bdx = cuda.blockDim.x
    bdy = cuda.blockDim.y
    x = bx * bdx + tx
    y = by * bdy + ty
    h, w, chan = src.shape
    if x >= w or y >= h:
        return
    for c in range(chan):
        tile[ty, tx, c] = src[y, x, c]
    cuda.syncthreads()
    for c in range(chan):
        out_of_range_pixel_value = 0
        pixel_value = 0
        for ky in range(-3, 4):
            for kx in range(-3, 4):
                ly = ty + ky
                lx = tx + kx
                if ly < 0 or ly >= bdy or lx < 0 or lx >= bdx or y + ky >= h or x + kx >= w:
                    out_of_range_pixel_value += matrix[ky + 3, kx + 3]
                else:
                    pixel_value += tile[ly, lx, c] * matrix[ky + 3, kx + 3]
        dst[y, x, c] = pixel_value // (matrix_sum - out_of_range_pixel_value)

img = plt.imread("img/img.jpg")
hostInput = np.zeros(img.shape, np.uint8)
devInput = cuda.to_device(img)
hostOutput = np.zeros(img.shape, np.uint8)
devOutput = cuda.to_device(hostOutput)
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))
devMatrix = cuda.to_device(np.array(GAUSSIAN_BLUR_MATRIX, dtype=np.int32))
t1 = time.time()
gaussian_blur_gpu_shared_memory[gridSize, blockSize](devInput, devOutput, devMatrix, GAUSSIAN_BLUR_MATRIX_SUM)
cuda.synchronize()
t2 = time.time()
hostOutput = devOutput.copy_to_host()
plt.imshow(hostOutput)
plt.imsave("img/img_blurred_gpu.jpg", hostOutput)
print(f"GPU time: {t2 - t1:.2f} seconds")

times = []
img = plt.imread("img/img.jpg")
hostInput = np.zeros(img.shape, np.uint8)
hostOutput = np.zeros(img.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))


for i in range(10):
    devInput = cuda.to_device(img)
    devOutput = cuda.to_device(hostOutput)
    devMatrix = cuda.to_device(np.array(GAUSSIAN_BLUR_MATRIX, dtype=np.int32))
    t1 = time.time()
    gaussian_blur_gpu[gridSize, blockSize](devInput, devOutput, devMatrix, GAUSSIAN_BLUR_MATRIX_SUM)
    cuda.synchronize()
    t2 = time.time()
    hostOutput = devOutput.copy_to_host()
    times.append(t2 - t1)

times_shared_memory = []
for i in range(10):
    devInput = cuda.to_device(img)
    devOutput = cuda.to_device(hostOutput)
    devMatrix = cuda.to_device(np.array(GAUSSIAN_BLUR_MATRIX, dtype=np.int32))
    t1 = time.time()
    gaussian_blur_gpu_shared_memory[gridSize, blockSize](devInput, devOutput, devMatrix, GAUSSIAN_BLUR_MATRIX_SUM)
    cuda.synchronize()
    t2 = time.time()
    hostOutput = devOutput.copy_to_host()
    times_shared_memory.append(t2 - t1)

averages = [np.mean(times), np.mean(times_shared_memory)]
plt.figure(figsize=(6, 5))
bars = plt.bar(['Without Shared Memory', 'With Shared Memory'], averages, color=['tab:blue', 'orange'])
plt.bar_label(bars, fmt='%.3f s')
plt.title('Average Execution Time (10 runs)')
plt.ylabel('Time (seconds)')
plt.tight_layout()
plt.savefig('img/execution_times.png')
