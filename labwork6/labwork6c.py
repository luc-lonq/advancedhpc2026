import math

from numba import cuda
from matplotlib import pyplot as plt
import numpy as np

@cuda.jit
def blend_two_images(src1, src2, dst, c):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if tidy >= src1.shape[0] or tidx >= src1.shape[1]:
        return
    for i in range(3):
        dst[tidy, tidx, i] = int(src1[tidy, tidx, i] * c + src2[tidy, tidx, i] * (1 - c))

img1 = plt.imread('img/blend1.jpg')
img2 = plt.imread('img/blend2.jpg')
devInput1 = cuda.to_device(img1)
devInput2 = cuda.to_device(img2)
devOutput = cuda.device_array(img1.shape, np.uint8)
blockSize = (32, 32)
gridSize = (math.ceil(img1.shape[1] / blockSize[0]), math.ceil(img1.shape[0] / blockSize[1]))

blend_two_images[gridSize, blockSize](devInput1, devInput2, devOutput, 0.5)
blended_img = devOutput.copy_to_host()
plt.imshow(blended_img)
plt.imsave('img/blend_0.5.png', blended_img)

blend_two_images[gridSize, blockSize](devInput1, devInput2, devOutput, 0.8)
blended_img = devOutput.copy_to_host()
plt.imshow(blended_img)
plt.imsave('img/blend_0.8.png', blended_img)

blend_two_images[gridSize, blockSize](devInput1, devInput2, devOutput, 0.2)
blended_img = devOutput.copy_to_host()
plt.imshow(blended_img)
plt.imsave('img/blend_0.2.png', blended_img)


avg_times_per_block_size = []
for blocksize in range (1, 33):
    times_for_block_size = []
    for _ in range(10):
        devInput1 = cuda.to_device(img1)
        devInput2 = cuda.to_device(img2)
        devOutput = cuda.device_array(img1.shape, np.uint8)
        blockSize = (blocksize, blocksize)
        gridSize = (math.ceil(img1.shape[1] / blockSize[0]), math.ceil(img1.shape[0] / blockSize[1]))
        start_time = cuda.event()
        end_time = cuda.event()
        start_time.record()
        blend_two_images[gridSize, blockSize](devInput1, devInput2, devOutput, 0.5)
        end_time.record()
        end_time.synchronize()
        elapsed_time = cuda.event_elapsed_time(start_time, end_time)
        times_for_block_size.append(elapsed_time)
    avg_times_per_block_size.append(np.mean(times_for_block_size))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 33), avg_times_per_block_size, marker='o')
plt.title('Blend two images - Avg execution time vs Block size')
plt.xlabel('Block size (n x n)')
plt.ylabel('Avg execution time (ms)')
plt.xticks(range(1, 33))
plt.grid()
plt.tight_layout()
plt.savefig('img/labwork6c.png')