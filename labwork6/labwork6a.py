import math

from numba import cuda
from matplotlib import pyplot as plt
import numpy as np

@cuda.jit
def image_binarization(src, dst, threshold):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy >= src.shape[0] or tidx >= src.shape[1]:
        return
    if src[tidy, tidx] > threshold:
        dst[tidy, tidx] = 255
    else:
        dst[tidy, tidx] = 0


img = plt.imread('img/windowsxp_grayscaled.jpg')
gray = np.ascontiguousarray(img[:, :, 0])
devInput = cuda.to_device(gray)
devOutput = cuda.device_array(gray.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(gray.shape[1] / blockSize[0]), math.ceil(gray.shape[0] / blockSize[1]))

image_binarization[gridSize, blockSize](devInput, devOutput, 128)

binarized_img = devOutput.copy_to_host()
plt.imshow(binarized_img, cmap='gray')
plt.imsave('img/binarized_img.png', binarized_img, cmap='gray')

avg_times_per_block_size = []
for blocksize in range (1, 33):
    times_for_block_size = []
    for _ in range(10):
        devInput = cuda.to_device(gray)
        devOutput = cuda.device_array(gray.shape, np.uint8)
        blockSize = (blocksize, blocksize)
        gridSize = (math.ceil(gray.shape[1] / blockSize[0]), math.ceil(gray.shape[0] / blockSize[1]))
        start_time = cuda.event()
        end_time = cuda.event()
        start_time.record()
        image_binarization[gridSize, blockSize](devInput, devOutput, 128)
        end_time.record()
        end_time.synchronize()
        elapsed_time = cuda.event_elapsed_time(start_time, end_time)
        times_for_block_size.append(elapsed_time)
    avg_times_per_block_size.append(np.mean(times_for_block_size))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 33), avg_times_per_block_size, marker='o')
plt.title('Binarization - Avg execution time vs Block size')
plt.xlabel('Block size (n x n)')
plt.ylabel('Avg execution time (ms)')
plt.xticks(range(1, 33))
plt.grid()
plt.tight_layout()
plt.savefig('img/labwork6a.png')