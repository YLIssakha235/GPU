import numpy as np

def make_points(n: int = 256, seed: int = 0):
    """
    Crée un ensemble simple de points avec des vitesses constantes.

    Lien avec le projet Cloth Simulation :
    - positions  : positions initiales des sommets (envoyées au GPU)
    - velocities : vitesses initiales des sommets (envoyées au GPU)
    - Chaque point correspondra plus tard à un sommet du tissu

    Format mémoire (aligné GPU) :
    - positions  : (n, 4) float32 → vec4<f32> en WGSL
    - velocities : (n, 4) float32 → vec4<f32> en WGSL
    """

    # Générateur pseudo-aléatoire (reproductibilité)
    rng = np.random.default_rng(seed)

    # Positions initiales dans un petit cube centré autour de l'origine
    pos = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)

    # Composante w = 1.0 (utile pour l'alignement mémoire et le rendu)
    pos_w = np.ones((n, 1), dtype=np.float32)

    # Positions finales envoyées au GPU
    positions = np.concatenate([pos, pos_w], axis=1)

    # Vitesses constantes initiales (petites valeurs pour la stabilité)
    vel = rng.uniform(-0.2, 0.2, size=(n, 3)).astype(np.float32)

    # Composante w = 0.0 (non utilisée pour les calculs)
    vel_w = np.zeros((n, 1), dtype=np.float32)

    # Vitesses finales envoyées au GPU
    velocities = np.concatenate([vel, vel_w], axis=1)

    return positions, velocities

def make_grid_cloth(width: int, height: int, rest: float = 0.1):
    """
    Crée une grille régulière WxH (cloth).
    - Chaque point a 4 voisins structuraux (gauche/droite/haut/bas) sauf aux bords.
    - rest = distance au repos entre deux voisins.

    Format mémoire :
    - positions  : (N,4) float32 -> vec4<f32> en WGSL
    - velocities : (N,4) float32 -> vec4<f32> en WGSL
    """
    N = width * height
    positions = np.zeros((N, 4), dtype=np.float32)
    velocities = np.zeros((N, 4), dtype=np.float32)

    # centrer le tissu autour de (0, y0, 0)
    ox = -0.5 * (width - 1) * rest
    oz = -0.5 * (height - 1) * rest
    y0 = 1.0

    idx = 0
    for j in range(height):
        for i in range(width):
            x = ox + i * rest
            z = oz + j * rest
            positions[idx] = (x, y0, z, 1.0)
            velocities[idx] = (0.0, 0.0, 0.0, 0.0)
            idx += 1

    return positions, velocities


def make_grid_indices(W: int, H: int) -> np.ndarray:
    """
    Retourne un index buffer (uint32) pour dessiner une grille W×H en triangles.
    2 triangles par quad => (W-1)*(H-1)*2 triangles => *3 indices.
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

            # Triangles CCW
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


print("data_init.py loaded.")