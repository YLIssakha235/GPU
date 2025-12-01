import numpy as np
import time

# Helper class to measure execution time
class Timer:
    def __init__(self, msg: str):
        self.msg = msg

    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"{self.msg}: {time.perf_counter() - self.start}")

# Input Data
n = 4194240
numpy_data = np.full(n, 3, dtype=np.int32)

with Timer("for in range loop"):
    res = []
    for x in numpy_data:
        res.append(x * x)

with Timer("numpy operation") as rep:
    res = numpy_data * numpy_data