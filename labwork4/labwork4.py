import math
import time
import numpy as np
from numba import cuda
from matplotlib import pyplot as plt
from utils import grayscaleGPU1D, grayscaleGPU2D

cuda.select_device(0)

img = plt.imread("img/windowsxp.jpg")
reshaped_img = img.reshape((img.shape[0] * img.shape[1], 3))

# grayscale 1D
hostInput = np.zeros(reshaped_img.shape, np.uint8)
devInput = cuda.to_device(reshaped_img)
hostOutput = np.zeros(reshaped_img.shape, np.uint8)
devOutput = cuda.to_device(hostOutput)
pixelCount = reshaped_img.shape[0]
blockSize = 64
gridSize = math.ceil(pixelCount / blockSize)
grayscaledGPUimg = grayscaleGPU1D[gridSize, blockSize](devInput, devOutput)

t1 = time.time()
grayscaledGPUimg = grayscaleGPU1D[gridSize, blockSize](devInput, devOutput)
cuda.synchronize()
t2 = time.time()
hostOutput = devOutput.copy_to_host()
grayscaledGPUimg = hostOutput.reshape((img.shape[0], img.shape[1], 3))
print(f"1D time: {t2 - t1:.6f} seconds")
plt.imsave("img/windowsxp_grayscaled_gpu.jpg", grayscaledGPUimg)

# grayscale 2D
hostInput = np.zeros(img.shape, np.uint8)
devInput = cuda.to_device(img)
hostOutput = np.zeros(img.shape, np.uint8)
devOutput = cuda.to_device(hostOutput)
pixelCount = img.shape[0]
blockSize = (32, 32)
gridSize = (math.ceil(img.shape[0] / 32), math.ceil(img.shape[1] / 32))

# first execution to compile the kernel function
grayscaledGPUimg = grayscaleGPU2D[gridSize, blockSize](devInput, devOutput)

t1 = time.time()
grayscaledGPUimg = grayscaleGPU2D[gridSize, blockSize](devInput, devOutput)
cuda.synchronize()
t2 = time.time()
hostOutput = devOutput.copy_to_host()
grayscaledGPUimg = hostOutput.reshape((img.shape[0], img.shape[1], 3))
print(f"2D time: {t2 - t1:.6f} seconds")
plt.imsave("img/windowsxp_grayscaled_gpu.jpg", grayscaledGPUimg)


blockLabels = []
avgTimes = []
for i in range(2, 33):
    times = []
    for j in range(10):
        hostInput = np.zeros(img.shape, np.uint8)
        devInput = cuda.to_device(hostInput)
        hostOutput = np.zeros(img.shape, np.uint8)
        devOutput = cuda.to_device(hostOutput)
        pixelCount = reshaped_img.shape[0]
        blockSize = (i, i)
        gridSize = (math.ceil(img.shape[0] / i), math.ceil(img.shape[1] / i))
        t1 = time.time()
        grayscaledGPUimg = grayscaleGPU2D[gridSize, blockSize](devInput, devOutput)
        cuda.synchronize()
        t2 = time.time()
        times.append(t2 - t1)
    blockLabels.append(f"{i}x{i}")
    avgTimes.append(np.mean(times))

plt.figure(figsize=(14, 5))
plt.bar(blockLabels, avgTimes)
plt.title("GPU Grayscale Average Execution Time (10 runs)")
plt.xlabel("Block size")
plt.ylabel("Average time (seconds)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("img/gpu_grayscale_execution_time_block_size.png")
plt.close()



# Comparison of 1D and 2D blocks and grids
times1D = []
times2D = []
for i in range(10):
    hostInput = np.zeros(reshaped_img.shape, np.uint8)
    devInput = cuda.to_device(reshaped_img)
    hostOutput = np.zeros(reshaped_img.shape, np.uint8)
    devOutput = cuda.to_device(hostOutput)
    pixelCount = reshaped_img.shape[0]
    blockSize = 256
    gridSize = math.ceil(pixelCount / blockSize)
    t1 = time.time()
    grayscaledGPUimg = grayscaleGPU1D[gridSize, blockSize](devInput, devOutput)
    cuda.synchronize()
    t2 = time.time()
    times1D.append(t2 - t1)

for i in range(10):
    hostInput = np.zeros(img.shape, np.uint8)
    devInput = cuda.to_device(hostInput)
    hostOutput = np.zeros(img.shape, np.uint8)
    devOutput = cuda.to_device(hostOutput)
    pixelCount = reshaped_img.shape[0]
    blockSize = (16, 16)
    gridSize = (math.ceil(img.shape[0] / 16), math.ceil(img.shape[1] / 16))
    t1 = time.time()
    grayscaledGPUimg = grayscaleGPU2D[gridSize, blockSize](devInput, devOutput)
    cuda.synchronize()
    t2 = time.time()
    times2D.append(t2 - t1)

plt.figure(figsize=(14, 5))
plt.bar(["1D", "2D"], [np.mean(times1D), np.mean(times2D)])
plt.title("GPU Grayscale Average Execution Time Comparison 1D vs 2D (10 runs)")
plt.xlabel("Dimension")
plt.ylabel("Average time (seconds)")
plt.tight_layout()
plt.savefig("img/gpu_grayscale_execution_time_comparison_1d_2d.png")
plt.close()
