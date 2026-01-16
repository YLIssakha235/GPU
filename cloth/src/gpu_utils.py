import numpy as np
import wgpu

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_storage_buffer(device: wgpu.GPUDevice, data: np.ndarray):
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    buf = device.create_buffer(size=data.nbytes, usage=usage)
    device.queue.write_buffer(buf, 0, data.tobytes())
    return buf

def create_vertex_storage_buffer(device: wgpu.GPUDevice, data: np.ndarray):
    """
    Buffer utilisé à la fois par:
    - compute shader (STORAGE)
    - render pipeline (VERTEX)
    """
    usage = (
        wgpu.BufferUsage.STORAGE |
        wgpu.BufferUsage.VERTEX |
        wgpu.BufferUsage.COPY_DST |
        wgpu.BufferUsage.COPY_SRC
    )
    buf = device.create_buffer(size=data.nbytes, usage=usage)
    device.queue.write_buffer(buf, 0, data.tobytes())
    return buf

def create_uniform_buffer(device: wgpu.GPUDevice, data_bytes: bytes):
    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    buf = device.create_buffer(size=len(data_bytes), usage=usage)
    device.queue.write_buffer(buf, 0, data_bytes)
    return buf

def create_index_buffer(device: wgpu.GPUDevice, indices: np.ndarray):
    usage = wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
    buf = device.create_buffer(size=indices.nbytes, usage=usage)
    device.queue.write_buffer(buf, 0, indices.tobytes())
    return buf
