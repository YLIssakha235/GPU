import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) 
    return v / n if n > 0 else v


def look_at(eye, target, up) -> np.ndarray:
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = normalize(target - eye)      # forward
    s = normalize(np.cross(f, up))   # right
    u = np.cross(s, f)               # corrected up

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    return M @ T


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    fovy = np.deg2rad(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2.0 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P
