# 🎓 GUIDE COMPLET DE PRÉPARATION - SIMULATION DE TISSU GPU

**Objectif : Cartonner ta présentation et obtenir 20/20 ! 🚀**

---

## 📋 CHECKLIST AVANT LA PRÉSENTATION

### ✅ Démo Technique
- [ ] Le projet se lance sans erreur (`python main.py`)
- [ ] Les contrôles souris/clavier fonctionnent
- [ ] Tu peux faire une démo en live (pause, reset, toggles)
- [ ] Tu as préparé 2-3 configs différentes (masse lourde, sphère déplacée, etc.)
- [ ] Tu sais où sont tous les fichiers importants

### ✅ Compréhension du Code
- [ ] Tu peux expliquer chaque fichier (`simulation.py`, `scene.py`, etc.)
- [ ] Tu comprends `self` et la POO
- [ ] Tu sais ce qu'est un compute shader vs render shader
- [ ] Tu peux expliquer les buffers (storage, vertex, index, uniform)
- [ ] Tu comprends le ping-pong (A ↔ B)

### ✅ Physique
- [ ] Tu peux expliquer le modèle masse-ressort
- [ ] Tu comprends la friction de Coulomb (statique + dynamique)
- [ ] Tu sais pourquoi on a des substeps
- [ ] Tu peux expliquer le problème de tunneling

---

## 🎤 QUESTIONS PROBABLES + TES RÉPONSES

---

### 🔴 QUESTIONS GÉNÉRALES

#### Q1 : "Explique-moi ton projet en 2 minutes"

**Ta réponse :**
> "J'ai développé une simulation de tissu en temps réel sur GPU avec Python et wgpu. Le tissu est modélisé comme un système masse-ressort avec 3 types de ressorts : structurels (horizontal/vertical), de cisaillement (diagonales) et de flexion (distance 2). 
>
> La physique tourne entièrement sur GPU via des **compute shaders** en WGSL. J'utilise un système de **ping-pong buffers** pour éviter les race conditions GPU. Le tissu peut entrer en collision avec une sphère et le sol, avec un modèle de friction de Coulomb (statique + dynamique).
>
> J'ai refactorisé le code en architecture modulaire avec 4 classes principales : `Simulation` (physique), `Scene` (rendu), `InputController` (contrôles), et des `Renderers` individuels pour chaque objet."

---

#### Q2 : "Pourquoi utiliser le GPU plutôt que le CPU ?"

**Ta réponse :**
> "Le GPU excelle dans le calcul parallèle. Chaque particule du tissu (144 dans mon cas : grille 12×12) peut être mise à jour **en parallèle** par un thread GPU différent. Sur CPU, il faudrait une boucle séquentielle qui traite chaque particule une par une.
>
> Avec les compute shaders, je lance `dispatch_workgroups(3)` où chaque workgroup de 64 threads traite plusieurs particules simultanément. C'est **beaucoup plus rapide** et ça scale mieux si j'augmente la résolution du tissu."

---

#### Q3 : "Qu'est-ce qu'un compute shader ?"

**Ta réponse :**
> "Un compute shader est un programme GPU généraliste (GPGPU) qui permet de faire des calculs arbitraires, contrairement aux vertex/fragment shaders qui sont limités au pipeline graphique.
>
> Dans mon projet, j'ai 3 compute shaders :
> 1. **`step2_structural_shear_bend.wgsl`** : calcule les forces des ressorts + gravité
> 2. **`step4_collision_friction.wgsl`** : gère les collisions sphère/sol + friction
> 3. **`compute_normals_grid.wgsl`** : recalcule les normales pour l'éclairage
>
> Ils lisent et écrivent dans des **storage buffers** (positions, vitesses)."

---

### 🔴 QUESTIONS SUR LA PHYSIQUE

#### Q4 : "Explique le modèle masse-ressort"

**Ta réponse :**
> "Chaque point du tissu est une particule de masse `m`. Les particules sont reliées par des ressorts virtuels qui exercent des forces de rappel selon la loi de Hooke : `F = -k * (longueur_actuelle - longueur_repos)`.
>
> J'ai 3 types de ressorts :
> - **Structurels** : relient les voisins horizontaux/verticaux (distance 1)
> - **Cisaillement** : relient les voisins diagonaux
> - **Flexion** : relient les voisins à distance 2 (pour éviter que le tissu se plie trop)
>
> À chaque frame, je calcule toutes les forces (gravité + ressorts), puis j'intègre avec Euler explicite : `v += F/m * dt` et `p += v * dt`. J'ajoute aussi un amortissement pour stabiliser."

---

#### Q5 : "C'est quoi la friction de Coulomb ?"

**Ta réponse :**
> "La friction de Coulomb est un modèle de friction qui distingue deux régimes :
>
> **Friction statique** : Si la force tangentielle est faible (`|F_tangent| < mu * |F_normal|`), la particule **colle** à la surface (vitesse tangentielle = 0).
>
> **Friction dynamique** : Sinon, la particule glisse et on applique une force de friction `F_friction = -mu * |F_normal| * direction_vitesse`.
>
> Dans mon shader `step4_collision_friction.wgsl`, je décompose la vitesse en composante normale (perpendiculaire à la surface) et tangentielle (parallèle), puis j'applique ce modèle. Le coefficient `MU` (0.6 par défaut) contrôle l'intensité de la friction."

---

#### Q6 : "Pourquoi tu as des substeps ?"

**Ta réponse :**
> "Les substeps augmentent la **stabilité** de la simulation. Si j'utilise un seul pas de temps `DT = 1/240s` par frame, les ressorts très raides peuvent causer des oscillations numériques ou des explosions.
>
> Avec `SUBSTEPS = 8`, je divise chaque frame en 8 micro-étapes de `dt_sub = DT/8`. Ça détecte mieux les collisions rapides et évite le **tunneling** (particules qui traversent la sphère).
>
> C'est un compromis : plus de substeps = plus précis, mais plus coûteux en calcul GPU."

---

#### Q7 : "C'est quoi le tunneling ? Comment tu le résous ?"

**Ta réponse :**
> "Le tunneling arrive quand une particule se déplace **trop vite** entre deux frames et saute à travers un objet sans détecter la collision.
>
> Exemple : si `MASS` est élevé, la gravité accélère beaucoup le tissu. Entre deux frames, une particule peut passer d'un côté de la sphère à l'autre.
>
> **Solutions** :
> 1. Augmenter `SUBSTEPS` (plus de vérifications par frame)
> 2. Augmenter `EPS` (tolérance collision, détecte "avant" le contact)
> 3. Réduire `DT` (pas de temps plus petit)
>
> Dans mon cas, passer de `SUBSTEPS = 8` à `SUBSTEPS = 20` résout le problème pour `MASS = 0.5`."

---

### 🔴 QUESTIONS SUR LE CODE

#### Q8 : "Qu'est-ce qu'un buffer en GPU ?"

**Ta réponse :**
> "Un buffer est une zone mémoire GPU. Il y a plusieurs types :
>
> - **Vertex Buffer** : positions des vertices (pour le rendu)
> - **Index Buffer** : indices des triangles/lignes
> - **Uniform Buffer** : petites données read-only (paramètres, matrices MVP)
> - **Storage Buffer** : grandes données read-write (positions, vitesses en compute)
>
> Dans mon projet, `pos_a` et `pos_b` sont des **storage buffers** pour le ping-pong. Je les crée avec :
> ```python
> self.pos_a = device.create_buffer_with_data(
>     data=positions_np.tobytes(),
>     usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX
> )
> ```
> `STORAGE` permet le R/W en compute, `VERTEX` permet de les utiliser en rendu."

---

#### Q9 : "C'est quoi le ping-pong ? Pourquoi ?"

**Ta réponse :**
> "Le ping-pong évite les **race conditions** en GPU. Si j'écris et lis dans le même buffer en parallèle, les threads GPU peuvent se marcher dessus.
>
> **Solution** : j'ai deux buffers `pos_a` et `pos_b`. À chaque étape :
> - Je **lis** dans A (positions actuelles)
> - Je **calcule** les nouvelles positions
> - J'**écris** dans B (nouvelles positions)
> - Puis je **swap** : A devient B, B devient A
>
> En code :
> ```python
> if ping:
>     # shader lit dans pos_a, écrit dans pos_b
> else:
>     # shader lit dans pos_b, écrit dans pos_a
> ping = not ping
> ```
> C'est le même principe qu'un **double buffering** en rendu."

---

#### Q10 : "Explique `self` en Python"

**Ta réponse :**
> "`self` représente **l'instance actuelle** d'une classe. C'est comme dire "moi-même".
>
> Exemple :
> ```python
> class Simulation:
>     def __init__(self, device):
>         self.G = -9.81      # "MA gravité"
>         self.MASS = 0.1     # "MA masse"
>     
>     def step(self):
>         print(self.G)  # Accède à MA gravité
> ```
>
> Avec les classes, je peux créer plusieurs simulations indépendantes :
> ```python
> sim1 = Simulation(device)
> sim1.MASS = 0.1
>
> sim2 = Simulation(device)
> sim2.MASS = 0.5  # Différent !
> ```
>
> Chaque instance a SES propres données. Sans classes (code global), c'est impossible."

---

#### Q11 : "Pourquoi tu as refactorisé en classes ?"

**Ta réponse :**
> "Mon ancien code avait tout dans un seul fichier `main.py` de 500+ lignes avec des variables globales partout. C'était difficile à maintenir.
>
> Avec la refactorisation :
> - **`Simulation`** : encapsule toute la physique (compute shaders, paramètres)
> - **`Scene`** : gère le rendu (caméra, géométrie, draw calls)
> - **`InputController`** : gère les entrées (souris, clavier)
> - **`Renderers`** : pipelines de rendu individuels
>
> **Avantages** :
> - Code modulaire et réutilisable
> - Séparation claire des responsabilités
> - Facile à débugger et étendre
> - Architecture professionnelle"

---

#### Q12 : "Comment tu gères la caméra ?"

**Ta réponse :**
> "J'utilise une **caméra orbit** qui tourne autour d'un point cible (le centre de la sphère).
>
> Elle a 3 paramètres :
> - `cam_yaw` : rotation horizontale (angle autour de Y)
> - `cam_pitch` : rotation verticale (angle d'élévation)
> - `cam_dist` : distance au centre
>
> Quand l'utilisateur drag la souris, je modifie `yaw` et `pitch`. Puis je calcule la position de la caméra en coordonnées sphériques :
> ```python
> eye_x = target_x + dist * sin(yaw) * cos(pitch)
> eye_y = target_y + dist * sin(pitch)
> eye_z = target_z + dist * cos(yaw) * cos(pitch)
> ```
>
> Ensuite je crée les matrices `view` (look_at) et `projection` (perspective), et je les multiplie pour avoir la matrice MVP finale."

---

### 🔴 QUESTIONS PIÈGES / AVANCÉES

#### Q13 : "Pourquoi pas utiliser Euler implicite au lieu d'explicite ?"

**Ta réponse :**
> "Euler implicite est plus stable mais **beaucoup plus coûteux**. Il faut résoudre un système d'équations linéaires à chaque étape (matrice sparse), ce qui est complexe à paralléliser sur GPU.
>
> Euler explicite (`v += F/m * dt`, `p += v * dt`) est simple et se parallélise parfaitement. Pour compenser l'instabilité, j'utilise :
> - Des substeps (divise `dt`)
> - De l'amortissement (`DAMPING = 0.995`)
> - Des ressorts pas trop raides
>
> Pour un projet éducatif, Euler explicite est un bon compromis **simplicité/performance**."

---

#### Q14 : "Tu pourrais ajouter du vent ? Comment ?"

**Ta réponse :**
> "Oui ! J'ajouterais une force de vent dans le compute shader `step2_structural_shear_bend.wgsl`.
>
> Exemple simple (vent constant) :
> ```wgsl
> let wind = vec3<f32>(5.0, 0.0, 2.0);  // Direction + intensité
> force += wind;
> ```
>
> Ou un vent turbulent (bruit de Perlin sur position + temps) :
> ```wgsl
> let noise = perlin_noise(position.xyz + time);
> let wind = vec3<f32>(noise * 10.0, 0.0, noise * 5.0);
> force += wind;
> ```
>
> Je passerais les paramètres du vent via un uniform buffer."

---

#### Q15 : "Comment tu testes les performances ?"

**Ta réponse :**
> "J'ai plusieurs métriques :
> - **FPS** : je compte les frames par seconde (devrait rester > 60)
> - **Temps GPU** : wgpu peut donner le temps d'exécution des compute passes
> - **Scalabilité** : je teste avec différentes tailles de grille (12×12, 20×20, 50×50)
>
> Actuellement avec une grille 12×12 (144 particules) et `SUBSTEPS=8`, je tourne à **60+ FPS** sur GPU moderne.
>
> Si je passe à 50×50 (2500 particules), ça descend mais reste temps réel. Le goulot d'étranglement est le nombre de substeps × nombre de particules."

---

#### Q16 : "Tu pourrais ajouter de l'auto-collision (tissu contre tissu) ?"

**Ta réponse :**
> "Oui, mais c'est **beaucoup plus complexe**. L'auto-collision nécessite de détecter quand une particule entre en collision avec **un triangle du tissu**.
>
> Approches possibles :
> 1. **Spatial hashing** : diviser l'espace en grille, tester seulement les particules proches
> 2. **BVH** (Bounding Volume Hierarchy) : accélération de structure en arbre
> 3. **Approche naïve** : tester toutes les paires (O(n²), trop lent)
>
> Sur GPU, le spatial hashing est faisable mais demande des compute shaders supplémentaires pour :
> - Construire la grille
> - Assigner particules aux cellules
> - Tester collisions dans chaque cellule
>
> C'est une extension intéressante mais hors scope pour ce projet."

---

## 🎯 STRATÉGIE DE PRÉSENTATION (15 min)

### 📌 INTRODUCTION (2 min)
1. "Bonjour, je vais vous présenter ma simulation de tissu GPU"
2. Montre la démo en live (pause, reset, toggles)
3. "J'ai utilisé Python + wgpu pour faire tourner la physique sur GPU"

### 📌 ARCHITECTURE (3 min)
1. Montre la structure du projet (dossiers)
2. Explique les 4 classes principales
3. "J'ai refactorisé pour avoir un code modulaire et maintenable"

### 📌 PHYSIQUE (4 min)
1. Explique le modèle masse-ressort (schéma si possible)
2. Montre les 3 types de ressorts
3. Explique collision + friction
4. Parle des substeps et du tunneling

### 📌 GPU (3 min)
1. Explique pourquoi GPU (parallélisme)
2. Montre les 3 compute shaders
3. Explique le ping-pong
4. Parle des buffers (storage, vertex, etc.)

### 📌 DÉMO INTERACTIVE (3 min)
1. Montre différentes masses (léger vs lourd)
2. Change la friction en live (si tu l'as codé)
3. Montre le tissu qui tombe au sol
4. Explique les paramètres clés

---

## 📚 FICHE DE RÉVISION EXPRESS

### Concepts Clés à Connaître

| Concept | Définition |
|---------|------------|
| **Compute Shader** | Programme GPU généraliste pour calculs parallèles |
| **Storage Buffer** | Buffer GPU lecture/écriture (R/W) |
| **Vertex Buffer** | Buffer contenant les positions des vertices |
| **Index Buffer** | Indices des triangles/lignes pour le rendu |
| **Uniform Buffer** | Petites données read-only (paramètres, MVP) |
| **Ping-pong** | Technique double-buffer pour éviter race conditions |
| **Substeps** | Division du pas de temps pour stabilité |
| **Tunneling** | Particule traverse objet (trop rapide) |
| **Friction Coulomb** | Modèle friction statique + dynamique |
| **MVP Matrix** | Model-View-Projection (transforme monde → écran) |
| **`self`** | Instance actuelle d'une classe (POO) |
| **Bind Group** | Collection de ressources liées pour les shaders |
| **WGSL** | WebGPU Shading Language (langage shaders) |
| **Race Condition** | Conflit accès concurrent mémoire GPU |

---

## 🗂️ AIDE-MÉMOIRE : FICHIERS & RESPONSABILITÉS

### Fichiers Principaux

**`main.py`** - Point d'entrée
- Lance juste `run_app()` depuis `src/app.py`
- **Ne touche jamais à ce fichier**

**`src/app.py`** - Chef d'orchestre
- Initialise GPU, canvas, contexte
- Crée `Simulation`, `Scene`, `InputController`
- Boucle de rendu : `sim.step()` → `sim.compute_normals()` → `scene.draw()`
- Gère le depth buffer dynamique

**`src/simulation.py`** - ⚙️ PHYSIQUE
- **TOUS les paramètres physiques** : G, MASS, K_STRUCT, K_SHEAR, K_BEND, DAMPING, DT, SUBSTEPS
- **Paramètres collision** : SPHERE_R, MU, EPS, BOUNCE, FLOOR_Y
- **Position sphère** : sphere_cx, sphere_cy, sphere_cz
- **Taille tissu** : W, H (ligne 50)
- Buffers GPU : pos_a/b, vel_a/b, normal_buf
- Compute pipelines : ressorts, collision, normales
- Méthodes : `step()`, `compute_normals()`, `reset()`

**`src/scene.py`** - 🎨 RENDU
- Caméra orbit : yaw, pitch, dist, target
- Paramètres caméra : ROT_SPEED, ZOOM_SPEED, limites pitch/dist
- Toggles affichage : show_cloth_surface, show_cloth_wire, show_sphere_surface, show_sphere_wire
- Géométrie : indices tissu (lignes + triangles), sphère (wire + surface)
- Renderers : cloth/sphere × wireframe/surface
- Méthodes : `compute_eye()`, `update_mvp()`, `draw()`

**`src/input_controller.py`** - 🎮 CONTRÔLES
- Handlers souris : drag (rotation), wheel (zoom)
- Handlers clavier : P (pause), R (reset), 1-4 (toggles), H (aide)
- Pour désactiver : commente `_hook_mouse()` ou `_hook_keyboard()`

**`src/data_init.py`** - 🔢 GÉNÉRATION MESH
- `make_grid_cloth()` : positions/vitesses initiales tissu
- `make_grid_indices()` : triangles tissu
- `make_grid_line_indices()` : lignes tissu (wireframe)
- `make_uv_sphere_wire()` : wireframe sphère
- `make_uv_sphere_triangles()` : surface sphère

**`src/camera.py`** - 📷 MATRICES
- `look_at(eye, center, up)` : matrice view
- `perspective(fov, aspect, near, far)` : matrice projection

**`src/gpu_utils.py`** - 🛠️ HELPERS
- `read_text(path)` : charge un fichier shader

---

### Shaders (WGSL)

**`step2_structural_shear_bend.wgsl`** - Ressorts + gravité
- Calcule forces : gravité + 3 types ressorts
- Intégration Euler : v += F/m * dt, p += v * dt
- Amortissement : v *= DAMPING
- Entrées : pos_in, vel_in, params (dt, g, k_struct, k_shear, k_bend, W, H)
- Sorties : pos_out, vel_out

**`step4_collision_friction.wgsl`** - Collision + friction
- Détecte collision sphère : dist < radius + eps
- Projette particule hors sphère : p = center + normal * (r + eps)
- Décompose vitesse : normale + tangentielle
- Friction Coulomb : statique (colle) ou dynamique (glisse)
- Collision sol : y < FLOOR_Y
- Entrées : pos_in, vel_in, params (dt, sphere_cx/cy/cz/r, mu, eps, floor_y)
- Sorties : pos_out, vel_out

**`compute_normals_grid.wgsl`** - Normales
- Recalcule normales pour éclairage
- Moyenne des normales des triangles adjacents
- Entrées : pos_in, params (W, H)
- Sorties : normal_buf

**`render_basic.wgsl`** - Wireframe
- Vertex shader : transforme positions via MVP
- Fragment shader : couleur unie (blanc)

**`render_lit.wgsl`** - Surface éclairée tissu
- Vertex shader : transforme positions + normales via MVP
- Fragment shader : Phong lighting (diffus + ambiant)
- Couleur : rose/rouge

**`render_sphere.wgsl`** - Wireframe sphère
- Vertex shader : applique transform sphère (center + radius) puis MVP
- Fragment shader : couleur jaune

**`render_sphere_lit.wgsl`** - Surface éclairée sphère
- Vertex shader : transform sphère + calcul normales
- Fragment shader : Phong lighting
- Couleur : gris

---

### Renderers (`src/renders/`)

**`cloth_renderer.py`** - Tissu wireframe
- Pipeline : lignes, blanc, depth read-only
- Shader : `render_basic.wgsl`
- Méthodes : `set_mvp()`, `encode()`

**`cloth_renderer_lit.py`** - Tissu surface
- Pipeline : triangles, éclairage, depth write
- Shader : `render_lit.wgsl`
- 2 vertex buffers : positions + normales
- Méthodes : `set_mvp()`, `encode()`

**`sphere_renderer.py`** - Sphère wireframe
- Pipeline : lignes, jaune, depth read-only
- Shader : `render_sphere.wgsl`
- Méthodes : `set_mvp()`, `set_sphere()`, `encode()`

**`sphere_renderer_lit.py`** - Sphère surface
- Pipeline : triangles, éclairage, depth write
- Shader : `render_sphere_lit.wgsl`
- Méthodes : `set_mvp()`, `set_sphere()`, `encode()`

---

## 🎯 MODIFICATIONS RAPIDES (où changer quoi)

### Physique
| Quoi | Fichier | Ligne | Paramètre |
|------|---------|-------|-----------|
| Gravité | `simulation.py` | 15 | `self.G = -9.81` |
| Masse particules | `simulation.py` | 23 | `self.MASS = 0.1` |
| Raideur ressorts | `simulation.py` | 17-19 | `K_STRUCT`, `K_SHEAR`, `K_BEND` |
| Amortissement | `simulation.py` | 21 | `self.DAMPING = 0.995` |
| Stabilité | `simulation.py` | 21 | `self.SUBSTEPS = 8` |
| Friction | `simulation.py` | 37 | `self.MU = 0.6` |

### Géométrie
| Quoi | Fichier | Ligne | Paramètre |
|------|---------|-------|-----------|
| Taille tissu | `simulation.py` + `scene.py` | 50 + 107 | `W, H = 12, 12` |
| Position sphère | `simulation.py` | 53 | `sphere_cx, cy, cz` |
| Rayon sphère | `simulation.py` | 34 | `SPHERE_R = 0.8` |
| Hauteur sol | `simulation.py` | 39 | `FLOOR_Y = 0.0` |
| Distance tissu/sphère | `simulation.py` | 56 | `cloth_y0 = ... + 0.10` |

### Caméra
| Quoi | Fichier | Ligne | Paramètre |
|------|---------|-------|-----------|
| Point visé | `scene.py` | 60 | `self.target` |
| Distance initiale | `scene.py` | 64 | `self.cam_dist = 4.5` |
| Angle initial | `scene.py` | 62-63 | `cam_yaw`, `cam_pitch` |
| Sensibilité souris | `scene.py` | 71 | `ROT_SPEED = 0.006` |
| Sensibilité zoom | `scene.py` | 72 | `ZOOM_SPEED = 0.15` |

### Affichage
| Quoi | Fichier | Ligne | Paramètre |
|------|---------|-------|-----------|
| Tissu surface | `scene.py` | 33 | `show_cloth_surface` |
| Tissu wireframe | `scene.py` | 34 | `show_cloth_wire` |
| Sphère surface | `scene.py` | 35 | `show_sphere_surface` |
| Sphère wireframe | `scene.py` | 36 | `show_sphere_wire` |
| Taille fenêtre | `app.py` | ~16 | `size=(900, 700)` |

### Contrôles
| Quoi | Fichier | Méthode | Action |
|------|---------|---------|--------|
| Désactiver souris | `input_controller.py` | `_hook_mouse()` | Vide la fonction |
| Désactiver clavier | `input_controller.py` | `_hook_keyboard()` | Vide la fonction |

---

## 💡 DERNIER CONSEIL

### Si le prof pose une question que tu connais pas :

❌ **NE DIS PAS** : 
- "Je sais pas"
- "J'ai copié ça d'Internet"
- "ChatGPT m'a aidé" (même si c'est vrai 😉)

✅ **DIS PLUTÔT** : 
> "C'est une bonne question ! Dans mon implémentation, j'ai utilisé [approche X]. Une amélioration possible serait [idée Y]. Je n'ai pas eu le temps de l'implémenter mais c'est dans mes notes d'extension."

**Ça montre que tu réfléchis et que tu as conscience des limites !**

---

## 🚀 DÉMOS À PRÉPARER

### Démo 1 : Configuration de base
- Tissu 12×12, MASS = 0.1, SUBSTEPS = 8
- Tissu tombe sur sphère, friction visible
- Montrer pause/resume, reset, toggles

### Démo 2 : Masse lourde (tunneling)
- MASS = 0.5, SUBSTEPS = 8
- Montrer le problème : tissu traverse
- Puis SUBSTEPS = 20 : problème résolu !

### Démo 3 : Tissu au sol
- Cacher sphère (touches 3 + 4)
- Tissu démarre haut (cloth_y0 = 2.5)
- Tombe et s'aplatit au sol

### Démo 4 : Paramètres extrêmes
- Gravité lunaire : G = -2.0
- Friction élevée : MU = 0.9
- Ressorts mous : K_STRUCT = 30.0

---

## ✅ CHECKLIST FINALE

- [ ] J'ai relu ce guide au moins 2 fois
- [ ] Je peux expliquer les 3 compute shaders sans regarder
- [ ] Je connais tous les paramètres de `simulation.py`
- [ ] Je peux dess