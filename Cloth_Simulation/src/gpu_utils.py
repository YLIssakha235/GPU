import numpy as np
import wgpu

def read_text(path: str) -> str:
    """Lit un fichier texte (shader WGSL)."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_storage_buffer(device: wgpu.GPUDevice, data: np.ndarray):
    """Crée un buffer STORAGE pour compute shaders."""
    usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    buf = device.create_buffer_with_data(data=data, usage=usage)
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
    buf = device.create_buffer_with_data(data=data, usage=usage)
    return buf

def create_uniform_buffer(device: wgpu.GPUDevice, data_bytes: bytes):
    """Crée un buffer UNIFORM pour paramètres."""
    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    buf = device.create_buffer_with_data(data=data_bytes, usage=usage)
    return buf

def create_index_buffer(device: wgpu.GPUDevice, indices: np.ndarray):
    """Crée un buffer INDEX pour le rendu."""
    usage = wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
    buf = device.create_buffer_with_data(data=indices, usage=usage)
    return buf

