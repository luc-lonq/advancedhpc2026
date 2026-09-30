import math
import time
import numpy as np
from numba import cuda
from matplotlib import pyplot as plt
from utils import grayscaleCPU, grayscaleGPU

cuda.select_device(0)

img = plt.imread("img/windowsxp.jpg")
reshaped_img = img.reshape((img.shape[0] * img.shape[1], 3))

# CPU grayscale
# t1 = time.time()
# grayscaledCPUimg = grayscaleCPU(reshaped_img.copy())
# t2 = time.time()
# print(f"CPU time: {t2 - t1:.6f} seconds")
# grayscaledCPUimg = grayscaledCPUimg.reshape((img.shape[0], img.shape[1], 3))
# plt.imshow(grayscaledCPUimg)
# plt.imsave("img/windowsxp_grayscaled_cpu.jpg", grayscaledCPUimg)
# plt.show()

# # CPU grayscale execution time plot
# times = []
# for i in range(5):
#     t1 = time.time()
#     grayscaledCPUimg = grayscaleCPU(reshaped_img.copy())
#     t2 = time.time()
#     times.append(t2 - t1)
#     plt.plot(times)
#     plt.title("CPU Grayscale Execution Time")
#     plt.xlabel("Execution")
#     plt.ylabel("Time (seconds)")
#     plt.savefig("cpu_grayscale_execution_time.png")

# # GPU grayscale
hostInput = np.zeros(reshaped_img.shape, np.uint8)
devInput = cuda.to_device(reshaped_img)
hostOutput = np.zeros(reshaped_img.shape, np.uint8)
devOutput = cuda.to_device(hostOutput)
pixelCount = reshaped_img.shape[0]
blockSize = 64
gridSize = math.ceil(pixelCount / blockSize)
t1 = time.time()
grayscaledGPUimg = grayscaleGPU[gridSize, blockSize](devInput, devOutput)
cuda.synchronize()
t2 = time.time()
hostOutput = devOutput.copy_to_host()
grayscaledGPUimg = hostOutput.reshape((img.shape[0], img.shape[1], 3))
print(f"GPU time: {t2 - t1:.6f} seconds")
plt.imshow(grayscaledGPUimg)
plt.imsave("img/windowsxp_grayscaled_gpu.jpg", grayscaledGPUimg)
plt.show()

# # GPU grayscale execution time plot
times = []
for i in range(100):
    hostInput = np.zeros(reshaped_img.shape, np.uint8)
    devInput = cuda.to_device(reshaped_img)
    hostOutput = np.zeros(reshaped_img.shape, np.uint8)
    devOutput = cuda.to_device(hostOutput)
    pixelCount = reshaped_img.shape[0]
    blockSize = 64
    gridSize = math.ceil(pixelCount / blockSize)
    t1 = time.time()
    grayscaledGPUimg = grayscaleGPU[gridSize, blockSize](devInput, devOutput)
    cuda.synchronize()
    t2 = time.time()
    times.append(t2 - t1)
plt.plot(times)
plt.title("GPU Grayscale Execution Time")
plt.xlabel("Execution")
plt.ylabel("Time (seconds)")
plt.savefig("img/gpu_grayscale_execution_time.png")

# GPU grayscale execution time plot, compare block sizes
plt.figure()
for blockSize in [32, 64, 128, 256]:
    times = []
    for i in range(100):
        hostInput = np.zeros(reshaped_img.shape, np.uint8)
        devInput = cuda.to_device(reshaped_img)
        hostOutput = np.zeros(reshaped_img.shape, np.uint8)
        devOutput = cuda.to_device(hostOutput)
        pixelCount = reshaped_img.shape[0]
        gridSize = math.ceil(pixelCount / blockSize)
        t1 = time.time()
        grayscaledGPUimg = grayscaleGPU[gridSize, blockSize](devInput, devOutput)
        cuda.synchronize()
        t2 = time.time()
        times.append(t2 - t1)
    plt.plot(times, label=f"Block size: {blockSize}")
plt.title("GPU Grayscale Execution Time")
plt.xlabel("Execution")
plt.ylabel("Time (seconds)")
plt.legend()
plt.savefig(f"img/gpu_grayscale_execution_time_block_size.png")


