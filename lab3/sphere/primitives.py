import math
import numpy as np


def sphere(radius=0.5, lat_segments=16, lon_segments=32):
    """Generate sphere vertex and index data.

    Vertex layout per-vertex: [x, y, z, nx, ny, nz, u, v]
    Returns (vertex_data, index_data) where:
      - vertex_data: flat numpy.float32 array of size (N*8,)
      - index_data: numpy.uint32 array of triangle indices

    Parameters:
      - radius: float, sphere radius
      - lat_segments: int, number of latitude bands (>=2)
      - lon_segments: int, number of longitude bands (>=3)
    """
    vertices = []
    indices = []

    for i in range(lat_segments + 1):
        theta = i * math.pi / lat_segments  # 0..pi
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for j in range(lon_segments + 1):
            phi = j * 2.0 * math.pi / lon_segments  # 0..2pi
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta

            nx, ny, nz = x, y, z
            u = 1.0 - (j / lon_segments)
            v = 1.0 - (i / lat_segments)

            vertices.append([radius * x, radius * y, radius * z, nx, ny, nz, u, v])

    # indices
    for i in range(lat_segments):
        for j in range(lon_segments):
            first = i * (lon_segments + 1) + j
            second = first + lon_segments + 1

            indices.append(first)
            indices.append(second)
            indices.append(first + 1)

            indices.append(second)
            indices.append(second + 1)
            indices.append(first + 1)

    vertex_data = np.array(vertices, dtype=np.float32)
    index_data = np.array(indices, dtype=np.uint32)

    return vertex_data.flatten(), index_data


__all__ = ["sphere"]

print("sphere defined")