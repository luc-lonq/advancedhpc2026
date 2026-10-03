import math

from numba import cuda
from matplotlib import pyplot as plt
import numba
import numpy as np

@cuda.jit
def to_grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy >= src.shape[0] or tidx >= src.shape[1]:
        return
    gray_value = (src[tidy, tidx, 0] + src[tidy, tidx, 1] + src[tidy, tidx, 2]) / 3
    dst[tidy, tidx] = gray_value


img = plt.imread('img/image.jpg')
devInput = cuda.to_device(img)
devOutput = cuda.device_array((img.shape[0], img.shape[1]), np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))
to_grayscale[gridSize, blockSize](devInput, devOutput)
gray_img = devOutput.copy_to_host()
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.imsave('img/image_grayscaled.png', gray_img, cmap='gray', vmin=0, vmax=255)

print(gray_img.min(), gray_img.max())

IMAGE_WIDTH = 5333
IMAGE_HEIGHT = 4000

REDUCE_BLOCK_SIZE = 1024

@cuda.jit
def compute_img_max_simple(src, dst):
    # shared memory declaration for caching block content
    cache = cuda.shared.array(REDUCE_BLOCK_SIZE, numba.uint8)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # copy local block from src to cache (in shared memory), 0 is neutral for max
    cache[localtid] = src[tid] + src[tid + cuda.blockDim.x]
    cuda.syncthreads()
    # reduction in cache
    s = 1
    while s < cuda.blockDim.x:
        if localtid % (s * 2) == 0:
            cache[localtid] = max(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s * 2

    if localtid == 0: dst[cuda.blockIdx.x] = cache[0]

# reduce repeatedly until a single value remains
gray_img_1D = gray_img.flatten()
devInput = cuda.to_device(gray_img_1D)

blockSize = REDUCE_BLOCK_SIZE
while devInput.shape[0] > 1:
    gridSize = math.ceil(devInput.shape[0] / blockSize)
    devOutput = cuda.device_array(gridSize, np.uint8)
    compute_img_max_simple[gridSize, blockSize](devInput, devOutput)
    devInput = devOutput
img_max = devInput.copy_to_host()[0]
print(img_max)


@cuda.jit
def compute_img_min_simple(src, dst):
    # shared memory declaration for caching block content
    cache = cuda.shared.array(REDUCE_BLOCK_SIZE, numba.uint8)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # copy local block from src to cache (in shared memory), 255 is neutral for min
    cache[localtid] = src[tid] + src[tid + cuda.blockDim.x]
    cuda.syncthreads()
    # reduction in cache
    s = 1
    while s < cuda.blockDim.x:
        if localtid % (s * 2) == 0:
            cache[localtid] = min(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s * 2

    if localtid == 0: dst[cuda.blockIdx.x] = cache[0]

devInput = cuda.to_device(gray_img_1D)

blockSize = REDUCE_BLOCK_SIZE
while devInput.shape[0] > 1:
    gridSize = math.ceil(devInput.shape[0] / blockSize)
    devOutput = cuda.device_array(gridSize, np.uint8)
    compute_img_min_simple[gridSize, blockSize](devInput, devOutput)
    devInput = devOutput
img_min = devInput.copy_to_host()[0]
print(img_min, gray_img_1D.min(), gray_img_1D.max())

@cuda.jit
def grayscale_stretching(src, dst, min_val, max_val):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy >= src.shape[0] or tidx >= src.shape[1]:
        return
    dst[tidy, tidx] = (src[tidy, tidx] - min_val) * 255 / (max_val - min_val)

devInput = cuda.to_device(gray_img)
devOutput = cuda.device_array(gray_img.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(gray_img.shape[1] / blockSize[0]), math.ceil(gray_img.shape[0] / blockSize[1]))
grayscale_stretching[gridSize, blockSize](devInput, devOutput, img_min, img_max)
stretched_img = devOutput.copy_to_host()
plt.imshow(stretched_img, cmap='gray', vmin=0, vmax=255)
plt.imsave('img/image_grayscaled_stretched.png', stretched_img, cmap='gray', vmin=0, vmax=255)

@cuda.jit
def compute_img_max_optimized(src, dst):
    cache = cuda.shared.array(REDUCE_BLOCK_SIZE, numba.uint8)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # copy local block from src to cache (in shared memory)
    cache[localtid] = src[tid] + src[tid + cuda.blockDim.x]
    cuda.syncthreads()
    # reduction in cache
    s = int(cuda.blockDim.x / 2)
    while s > 0:
        if (localtid < s):
            cache[localtid] = max(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s // 2
    # only first thread writes back to dst
    if localtid == 0: dst[cuda.blockIdx.x] = cache[0]

@cuda.jit
def compute_img_min_optimized(src, dst):
    cache = cuda.shared.array(REDUCE_BLOCK_SIZE, numba.uint8)
    localtid = cuda.threadIdx.x
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    # copy local block from src to cache (in shared memory)
    cache[localtid] = src[tid] + src[tid + cuda.blockDim.x]
    cuda.syncthreads()
    # reduction in cache
    s = int(cuda.blockDim.x / 2)
    while s > 0:
        if (localtid < s):
            cache[localtid] = min(cache[localtid], cache[localtid + s])
        cuda.syncthreads()
        s = s // 2
    # only first thread writes back to dst
    if localtid == 0: dst[cuda.blockIdx.x] = cache[0]


simple_times = []
optimized_times = []


devInput = cuda.to_device(gray_img)
devOutput = cuda.device_array(gray_img.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(gray_img.shape[1] / blockSize[0]), math.ceil(gray_img.shape[0] / blockSize[1]))
grayscale_stretching[gridSize, blockSize](devInput, devOutput, img_min, img_max)
stretched_img = devOutput.copy_to_host()
plt.imshow(stretched_img, cmap='gray', vmin=0, vmax=255)
plt.imsave('img/image_grayscaled_optimized.png', stretched_img, cmap='gray', vmin=0, vmax=255)



for _ in range(10):
    devInput = cuda.to_device(gray_img_1D)
    blockSize = REDUCE_BLOCK_SIZE
    start_time = cuda.event()
    end_time = cuda.event()
    start_time.record()
    while devInput.shape[0] > 1:
        gridSize = math.ceil(devInput.shape[0] / blockSize)
        devOutput = cuda.device_array(gridSize, np.uint8)
        compute_img_max_simple[gridSize, blockSize](devInput, devOutput)
        devInput = devOutput
    end_time.record()
    end_time.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_time, end_time)
    simple_times.append(elapsed_time)

    devInput = cuda.to_device(gray_img_1D)
    blockSize = REDUCE_BLOCK_SIZE
    start_time = cuda.event()
    end_time = cuda.event()
    start_time.record()
    while devInput.shape[0] > 1:
        gridSize = math.ceil(devInput.shape[0] / blockSize)
        devOutput = cuda.device_array(gridSize, np.uint8)
        compute_img_max_optimized[gridSize, blockSize](devInput, devOutput)
        devInput = devOutput
    end_time.record()
    end_time.synchronize()
    elapsed_time = cuda.event_elapsed_time(start_time, end_time)
    optimized_times.append(elapsed_time)

simple_avg_time = np.mean(simple_times)
optimized_avg_time = np.mean(optimized_times)
plt.figure(figsize=(10, 6))
plt.bar(['Simple', 'Optimized'], [simple_avg_time, optimized_avg_time], color=['blue', 'orange'])
plt.title('Average Execution Time for Image Max Computation')
plt.ylabel('Time (ms)')
plt.tight_layout()
plt.savefig('img/labwork7.png')