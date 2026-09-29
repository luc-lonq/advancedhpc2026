from numba import cuda

print(cuda.detect())

cuda.select_device(0)

device = cuda.get_current_device()

# print name
print(device.name)

# multiprocessor count
multiprocessor_count = device.MULTIPROCESSOR_COUNT
print(f"Multiprocessor count: {multiprocessor_count}")

# core count
core_count = device.MULTIPROCESSOR_COUNT * 128 # Compute capability 8.6 => 128 cores per multiprocessor
print(f"Core count: {core_count}")

# memory size
_, total = cuda.current_context().get_memory_info()
print(f"Total memory: {total // (1024 ** 2)} MB")
