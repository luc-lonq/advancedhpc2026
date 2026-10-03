import math

from numba import cuda
from matplotlib import pyplot as plt
import numpy as np

@cuda.jit
def brightness_control(src, dst, value):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy >= src.shape[0] or tidx >= src.shape[1]:
        return
    for i in range(3):
        dst[tidy, tidx, i] = min(max(src[tidy, tidx, i] + value, 0), 255)

img = plt.imread('img/windowsxp.jpg')
devInput = cuda.to_device(img)
devOutput = cuda.device_array(img.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))

brightness_control[gridSize, blockSize](devInput, devOutput, 100)
add_100 = devOutput.copy_to_host()
plt.imshow(add_100)
plt.imsave('img/add_100.png', add_100)

brightness_control[gridSize, blockSize](devInput, devOutput, -100)
remove_100 = devOutput.copy_to_host()
plt.imshow(remove_100)
plt.imsave('img/remove_100.png', remove_100)


avg_times_per_block_size = []
for blocksize in range (1, 33):
    times_for_block_size = []
    for _ in range(10):
        devInput = cuda.to_device(img)
        devOutput = cuda.device_array(img.shape, np.uint8)
        blockSize = (blocksize, blocksize)
        gridSize = (math.ceil(img.shape[1] / blockSize[0]), math.ceil(img.shape[0] / blockSize[1]))
        start_time = cuda.event()
        end_time = cuda.event()
        start_time.record()
        brightness_control[gridSize, blockSize](devInput, devOutput, 100)
        end_time.record()
        end_time.synchronize()
        elapsed_time = cuda.event_elapsed_time(start_time, end_time)
        times_for_block_size.append(elapsed_time)
    avg_times_per_block_size.append(np.mean(times_for_block_size))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 33), avg_times_per_block_size, marker='o')
plt.title('Brightness control - Avg execution time vs Block size')
plt.xlabel('Block size (n x n)')
plt.ylabel('Avg execution time (ms)')
plt.xticks(range(1, 33))
plt.grid()
plt.tight_layout()
plt.savefig('img/labwork6b.png')