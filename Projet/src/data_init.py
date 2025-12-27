import numpy as np

# ============================================================
# Données simples (debug)
# ============================================================

def make_points(n: int = 256, seed: int = 0):
    """
    Crée un ensemble simple de points avec des vitesses constantes.

    Format mémoire (aligné GPU) :
    - positions  : (n, 4) float32 → vec4<f32> en WGSL
    - velocities : (n, 4) float32 → vec4<f32> en WGSL
    """
    rng = np.random.default_rng(seed)

    pos = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    pos_w = np.ones((n, 1), dtype=np.float32)
    positions = np.concatenate([pos, pos_w], axis=1)

    vel = rng.uniform(-0.2, 0.2, size=(n, 3)).astype(np.float32)
    vel_w = np.zeros((n, 1), dtype=np.float32)
    velocities = np.concatenate([vel, vel_w], axis=1)

    return positions, velocities


# ============================================================
# Tissu : grille régulière
# ============================================================

def make_grid_cloth(
    width: int,
    height: int,
    rest: float = 0.1,
    y0: float = 1.5,
    cx: float = 0.0,
    cz: float = 0.0
):
    """
    Crée une grille régulière WxH (cloth), centrée autour de (cx, y0, cz).

    - rest = distance au repos entre voisins (pas de la grille)
    - y0   = hauteur initiale (important pour tomber sur la sphère)
    - cx,cz = décalage du tissu en X/Z (utile pour casser la symétrie)

    Format mémoire :
    - positions  : (N,4) float32 -> vec4<f32> en WGSL
    - velocities : (N,4) float32 -> vec4<f32> en WGSL
    """
    N = width * height
    positions = np.zeros((N, 4), dtype=np.float32)
    velocities = np.zeros((N, 4), dtype=np.float32)

    # On centre la grille autour de 0 en X/Z puis on ajoute cx/cz
    ox = -0.5 * (width - 1) * rest
    oz = -0.5 * (height - 1) * rest

    idx = 0
    for j in range(height):
        for i in range(width):
            x = ox + i * rest
            z = oz + j * rest
            positions[idx] = (x + cx, y0, z + cz, 1.0)
            velocities[idx] = (0.0, 0.0, 0.0, 0.0)
            idx += 1

    return positions, velocities


def make_grid_indices(W: int, H: int) -> np.ndarray:
    """
    Index buffer (uint32) pour dessiner une grille W×H en triangles.
    2 triangles par quad => (W-1)*(H-1)*2 triangles => *3 indices
    """
    def idx(x, y):
        return x + y * W

    indices = []
    for y in range(H - 1):
        for x in range(W - 1):
            i00 = idx(x, y)
            i10 = idx(x + 1, y)
            i01 = idx(x, y + 1)
            i11 = idx(x + 1, y + 1)

            # Triangles (ordre CCW)
            indices += [i00, i10, i01]
            indices += [i10, i11, i01]

    return np.array(indices, dtype=np.uint32)


def make_grid_line_indices(W: int, H: int, diagonals: bool = False) -> np.ndarray:
    """
    Index buffer pour affichage wireframe (lines).
    - Toujours: arêtes horizontales + verticales
    - Optionnel: diagonales (triangulation)
    """
    def idx(x, y):
        return x + y * W

    lines = []

    # horizontales
    for y in range(H):
        for x in range(W - 1):
            lines += [idx(x, y), idx(x + 1, y)]

    # verticales
    for y in range(H - 1):
        for x in range(W):
            lines += [idx(x, y), idx(x, y + 1)]

    # diagonales (optionnel) : une diagonale par quad
    if diagonals:
        for y in range(H - 1):
            for x in range(W - 1):
                i00 = idx(x, y)
                i11 = idx(x + 1, y + 1)
                lines += [i00, i11]

    return np.array(lines, dtype=np.uint32)


# ============================================================
# Sphère : wireframe (lignes)
# ============================================================

def make_sphere_wireframe(radius=1.0, lat=12, lon=24):
    """
    Génère un mesh wireframe de sphère (positions + indices lignes).
    - radius : rayon
    - lat    : subdivisions latitude
    - lon    : subdivisions longitude
    """
    positions = []
    indices = []

    for i in range(lat + 1):
        theta = np.pi * i / lat
        y = radius * np.cos(theta)
        r = radius * np.sin(theta)

        for j in range(lon):
            phi = 2 * np.pi * j / lon
            x = r * np.cos(phi)
            z = r * np.sin(phi)
            positions.append([x, y, z, 1.0])

    def idx(i, j):
        return i * lon + (j % lon)

    for i in range(lat):
        for j in range(lon):
            indices += [idx(i, j), idx(i, j + 1)]
            indices += [idx(i, j), idx(i + 1, j)]

    return (
        np.array(positions, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )


def make_uv_sphere_wire(stacks: int = 12, slices: int = 24):
    """
    Retourne un mesh sphère wireframe centré en (0,0,0), rayon 1.
    - positions: (Ns,4) float32
    - indices_lines: (Ms,) uint32 pour PrimitiveTopology.line_list
    """
    verts = []

    for i in range(stacks + 1):
        v = i / stacks
        phi = np.pi * v
        y = np.cos(phi)
        r = np.sin(phi)
        for j in range(slices):
            u = j / slices
            theta = 2 * np.pi * u
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            verts.append([x, y, z, 1.0])

    positions = np.array(verts, dtype=np.float32)

    def vid(i, j):
        return i * slices + (j % slices)

    lines = []
    # anneaux
    for i in range(stacks + 1):
        for j in range(slices):
            lines += [vid(i, j), vid(i, j + 1)]
    # méridiens
    for i in range(stacks):
        for j in range(slices):
            lines += [vid(i, j), vid(i + 1, j)]

    indices = np.array(lines, dtype=np.uint32)
    return positions, indices


# ============================================================
# Sphère : TRIANGLES (surface)  ✅ ajout pour SphereRendererLit
# ============================================================

def make_uv_sphere_triangles(stacks: int = 16, slices: int = 32):
    """
    Retourne un mesh sphère en TRIANGLES, centré (0,0,0), rayon 1.

    - positions: (Nv,4) float32  (vec4<f32>)
    - indices_triangles: (Nt,) uint32  pour PrimitiveTopology.triangle_list

    Remarque :
    - On utilise (slices + 1) sommets par anneau pour éviter la couture UV.
    - Normales = position.xyz normalisée (dans le shader, pas besoin de buffer normal).
    """
    verts = []

    for i in range(stacks + 1):
        v = i / stacks
        phi = np.pi * v
        y = np.cos(phi)
        r = np.sin(phi)

        for j in range(slices + 1):
            u = j / slices
            theta = 2 * np.pi * u
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            verts.append([x, y, z, 1.0])

    positions = np.array(verts, dtype=np.float32)

    idx = []
    stride = slices + 1
    for i in range(stacks):
        for j in range(slices):
            a = i * stride + j
            b = a + 1
            c = a + stride
            d = c + 1

            # 2 triangles par quad
            idx += [a, c, b]
            idx += [b, c, d]

    indices = np.array(idx, dtype=np.uint32)
    return positions, indices


print("data_init.py loaded.")
