import numpy as np
import pygame

# ==== Paramètres de la grille ====
GRID_WIDTH  = 100   # nombre de cellules en largeur
GRID_HEIGHT = 80    # nombre de cellules en hauteur
CELL_SIZE   = 8     # taille d'une cellule en pixels

SCREEN_WIDTH  = GRID_WIDTH  * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

FPS = 10 # génere 10 images par seconde

# ==== Logique Game of Life (CPU) ====

def random_grid(h: int, w: int, p_alive: float = 0.2) -> np.ndarray:
    """Grille aléatoire de 0 (mort) et 1 (vivant)."""
    return (np.random.rand(h, w) < p_alive).astype(np.uint8)

def step_cpu(grid: np.ndarray) -> np.ndarray:
    """Calcule la génération suivante (version CPU avec numpy)."""
    # on compte les voisins avec des décalages (bords en mode "torus")
    neighbors = (
        np.roll(grid,  1, 0) + np.roll(grid, -1, 0) +  # haut / bas
        np.roll(grid,  1, 1) + np.roll(grid, -1, 1) +  # gauche / droite
        np.roll(np.roll(grid, 1, 0),  1, 1) +          # haut-gauche
        np.roll(np.roll(grid, 1, 0), -1, 1) +          # haut-droite
        np.roll(np.roll(grid,-1, 0),  1, 1) +          # bas-gauche
        np.roll(np.roll(grid,-1, 0), -1, 1)            # bas-droite
    )

    # règles de Conway :
    # - cellule vivante + 2 ou 3 voisins → survit
    # - cellule morte  + 3 voisins → naît
    survive = (grid == 1) & ((neighbors == 2) | (neighbors == 3))
    birth   = (grid == 0) & (neighbors == 3)
    new_grid = np.where(survive | birth, 1, 0).astype(np.uint8)
    return new_grid

# ==== Affichage pygame ====

def draw_grid(screen, grid: np.ndarray):
    """Dessine la grille sur la fenêtre pygame."""
    screen.fill((0, 0, 0))  # fond noir
    alive_color = (0, 200, 0)

    h, w = grid.shape
    for y in range(h):
        for x in range(w):
            if grid[y, x] == 1:
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, alive_color, rect)

    pygame.display.flip()

# ==== Boucle principale ====

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Conway's Game of Life - CPU + pygame")
    clock = pygame.time.Clock()

    grid = random_grid(GRID_HEIGHT, GRID_WIDTH, p_alive=0.25)

    running = True
    paused = False

    while running:
        # --- gestion des événements ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused   # pause / reprise avec espace
                elif event.key == pygame.K_r:
                    grid = random_grid(GRID_HEIGHT, GRID_WIDTH, 0.25)

        # --- mise à jour de la grille ---
        if not paused:
            grid = step_cpu(grid)

        # --- dessin ---
        draw_grid(screen, grid)

        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
