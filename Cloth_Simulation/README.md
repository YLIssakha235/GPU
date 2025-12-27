# Simulation de Tissu - Python + wgpu (Compute + Rendu GPU)

Simulation physique de tissu en temps réel entièrement sur GPU avec Python et wgpu.

**Fonctionnalités :**
- Système masse-ressort avec ressorts structurels, de cisaillement et de flexion
- Collision avec sphère et sol + friction de Coulomb
- Shaders GPU pour la physique (WGSL)
- Caméra orbitale contrôlée à la souris
- Architecture refactorisée : `Simulation` / `Scene` / `Renderers` / `InputController`

---

## Démarrage Rapide

### Prérequis
- Python 3.10+
- GPU compatible wgpu

### Installation
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Lancer
```bash
python main.py
```

---

## Contrôles

### Caméra (Orbite)
- **Glisser souris** : Rotation caméra (yaw/pitch)
- **Molette souris** : Zoom avant/arrière

### Clavier
- **P** : Pause/Reprise simulation
- **R** : Réinitialiser le tissu
- **1** : Afficher/masquer tissu surface
- **2** : Afficher/masquer tissu wireframe
- **3** : Afficher/masquer sphère surface
- **4** : Afficher/masquer sphère wireframe
- **H** : Afficher l'aide

---

## Structure du Projet

```
Cloth_Simulation/
├── main.py                    # Point d'entrée
├── shaders/                   # Programmes GPU WGSL
│   ├── step2_structural_shear_bend.wgsl    # Ressorts + gravité
│   ├── step4_collision_friction.wgsl       # Collision + friction
│   ├── compute_normals_grid.wgsl           # Calcul des normales
│   ├── render_basic.wgsl                   # Rendu wireframe
│   ├── render_lit.wgsl                     # Rendu surface éclairée
│   ├── render_sphere.wgsl                  # Wireframe sphère
│   └── render_sphere_lit.wgsl              # Surface sphère
└── src/
    ├── __init__.py
    ├── app.py                 # Boucle principale + init GPU
    ├── simulation.py          # Physique (pipelines compute)
    ├── scene.py               # Rendu (caméra + géométrie)
    ├── input_controller.py    # Gestion souris + clavier
    ├── data_init.py           # Génération mesh (CPU)
    ├── camera.py              # Matrices view/projection
    ├── gpu_utils.py           # Utilitaires
    └── renders/
        ├── __init__.py
        ├── cloth_renderer.py          # Tissu wireframe
        ├── cloth_renderer_lit.py      # Tissu surface (éclairé)
        ├── sphere_renderer.py         # Sphère wireframe
        └── sphere_renderer_lit.py     # Sphère surface (éclairée)
```

---

## Vue Technique

### Fichiers Shader (WGSL)
- **Vertex Shader** : Exécuté une fois par vertex. Transforme les positions en clip space via la matrice MVP.
- **Fragment Shader** : Exécuté une fois par pixel. Calcule la couleur finale (éclairage ou couleur unie).
- **Compute Shader** : Calculs génériques GPU (lecture/écriture dans les buffers pour la physique).

### Buffers
- **Storage Buffers** (R/W) : Positions & vitesses (ping-pong A/B)
- **Vertex Buffers** : Positions pour le rendu
- **Index Buffers** : Triangles (surface) / lignes (wireframe)
- **Uniform Buffers** : Paramètres physiques (dt, k, g, mu...) + matrice MVP caméra

### Bind Groups
Collection de ressources (buffers) liées ensemble pour l'accès dans les shaders :
- `pos_in`, `vel_in` (lecture seule)
- `pos_out`, `vel_out` (écriture)
- `params` (uniform)

---

## Simulation Physique

### Modèle Masse-Ressort
Chaque particule du tissu subit :
- **Gravité** : `F = m * g`
- **Forces de ressorts** :
  - Structurels : voisins horizontaux/verticaux
  - Cisaillement : voisins diagonaux
  - Flexion : voisins à distance 2
- **Intégration** : Euler explicite + amortissement

**Compute Shader** : `step2_structural_shear_bend.wgsl`

### Détection de Collision
- **Sphère** : Projette les particules à l'extérieur de la surface de la sphère
- **Sol** : Empêche les particules de tomber sous `FLOOR_Y`
- **Friction** : Modèle de Coulomb (statique + dynamique)
  - Statique : la particule "colle" si la force tangentielle est faible
  - Dynamique : la particule glisse avec coefficient de friction `MU`

**Compute Shader** : `step4_collision_friction.wgsl`

### Calcul des Normales
Recalcule les normales par vertex pour l'éclairage de la grille du tissu.

**Compute Shader** : `compute_normals_grid.wgsl`

---

## Paramètres Clés

Dans `src/simulation.py` :

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `G` | Gravité (m/s²) | -9.81 |
| `MASS` | Masse des particules | 0.1 |
| `K_STRUCT` | Raideur ressorts structurels | 60.0 |
| `K_SHEAR` | Raideur ressorts cisaillement | 80.0 |
| `K_BEND` | Raideur ressorts flexion | 300.0 |
| `DAMPING` | Amortissement vitesse | 0.995 |
| `SUBSTEPS` | Sous-étapes physique par frame | 8 |
| `MU` | Coefficient de friction | 0.6 |
| `EPS` | Tolérance collision | 0.004 |
| `SPHERE_R` | Rayon sphère | 0.8 |
| `FLOOR_Y` | Hauteur du sol | 0.0 |

**Astuce** : Pour un tissu plus lourd (`MASS > 0.5`), augmenter `SUBSTEPS` à 16-32 pour éviter la traversée (tunneling).

---

## Guide de Personnalisation

### Changer la Taille du Tissu
**Fichiers** : `src/simulation.py` (ligne 50), `src/scene.py` (ligne 107)
```python
self.W, self.H = 20, 20  # Grille 20x20 (au lieu de 12x12)
```

### Changer Position/Taille de la Sphère
**Fichier** : `src/simulation.py`
```python
self.sphere_cx, self.sphere_cy, self.sphere_cz = 0.35, 1.0, 0.0  # Position
self.SPHERE_R = 1.2  # Rayon
```

### Changer la Physique
**Fichier** : `src/simulation.py`
```python
self.G = -5.0          # Gravité lunaire
self.K_STRUCT = 100.0  # Ressorts plus rigides
self.MU = 0.9          # Plus de friction
```

### Désactiver les Contrôles (pour présentation)
**Fichier** : `src/input_controller.py`
```python
def _hook_mouse(self):
    pass  # Désactive souris

def _hook_keyboard(self):
    pass  # Désactive clavier
```

---

## Liens & Ressources

### wgpu & WebGPU
- [Documentation wgpu-py](https://wgpu-py.readthedocs.io/)
- [Spécification WebGPU](https://www.w3.org/TR/webgpu/)
- [Spécification WGSL](https://www.w3.org/TR/WGSL/)

### Tutoriels
- [Guide wgpu Bootstrap](https://github.com/gfx-rs/wgpu)

### Physique & Graphisme
- [Simulation Tissu Masse-Ressort](https://graphics.stanford.edu/~mdfisher/cloth.html)
- [Modèle de Friction de Coulomb](https://fr.wikipedia.org/wiki/Frottement#Loi_de_Coulomb)
- [Génération Sphère UV](https://songho.ca/opengl/gl_sphere.html)

### Python & GPU
- [RenderCanvas](https://github.com/pygfx/rendercanvas)
- [NumPy](https://numpy.org/)

---

## Notes d'Architecture

### Conception Orientée Objet
Le code refactorisé utilise la **Programmation Orientée Objet (POO)** :
- `ClothSimulation` : Encapsule toute la physique (compute shaders, buffers, paramètres)
- `Scene` : Gère le rendu (caméra, géométrie, appels de dessin)
- `InputController` : Gère les entrées utilisateur (souris, clavier)
- `Renderers` : Pipelines de rendu individuels (tissu/sphère, wireframe/surface)

**Avantages** :
- Modulaire et réutilisable
- Séparation claire des responsabilités
- Facile à étendre (ajouter objets, shaders, etc.)

### `self` dans les Classes Python
`self` fait référence à l'instance actuelle d'une classe :
```python
class Simulation:
    def __init__(self, device):
        self.G = -9.81      # "MA gravité"
        self.device = device  # "MON device"
    
    def step(self):
        print(self.G)  # Accède à MA gravité
```

Cela permet d'avoir plusieurs simulations indépendantes avec différents paramètres.

---

## Dépannage

### Le Tissu Traverse la Sphère
**Problème** : En augmentant `MASS`, les particules traversent la sphère (tunneling).

**Solution** : Augmenter `SUBSTEPS` dans `src/simulation.py` :
```python
self.SUBSTEPS = 20  # Au lieu de 8
```

### Problèmes de Performance
- Réduire `SUBSTEPS` (moins précis mais plus rapide)
- Réduire la taille de la grille (`W`, `H`)
- Désactiver le rendu wireframe (touches `2` et `4`)

---

## Licence
Ce projet est à but éducatif.

---

**Fait avec Python 🐍 + wgpu 🎮 + WGSL ✨**