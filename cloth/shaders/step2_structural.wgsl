// Étape 2A — Ressorts structuraux (voisins directs) + gravité
// Double buffering : pos_in/vel_in → pos_out/vel_out
//
// Principe (par sommet i) :
//  1) calculer la force totale F (ressorts + gravité)
//  2) a = F / m
//  3) v_new = v + a * dt
//  4) amortissement : v_new *= damping
//  5) p_new = p + v_new * dt
//
// Lien direct avec le projet Cloth Simulation :
// - Chaque sommet du tissu est une particule
// - Les ressorts structuraux relient les voisins (gauche/droite/haut/bas)
// - PINNING : on fixe 2 coins pour voir le tissu "pendre"

struct Params {
    dt: f32,       // pas de temps (Δt)
    g: f32,        // accélération gravitationnelle
    k: f32,        // constante de raideur (Hooke)
    rest: f32,     // longueur au repos du ressort structural (L0)

    mass: f32,     // masse d’un sommet
    damping: f32,  // amortissement simple (multiplie la vitesse)
    _pad0: f32,
    _pad1: f32,

    width: u32,    // largeur de la grille (nb de points en X)
    height: u32,   // hauteur de la grille (nb de points en Y)
    n: u32,        // nombre total de points = width * height
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>  pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : Params;

// Conversion (x,y) → index linéaire i
fn idx_of(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Ajoute la force d’un ressort entre p (point courant) et q (voisin)
fn add_spring_force(p: vec3<f32>, q: vec3<f32>, force: ptr<function, vec3<f32>>) {
    let d = q - p;          // vecteur entre les deux points
    let L = length(d);      // longueur actuelle du ressort

    // Sécurité : éviter division par zéro si deux points sont quasi confondus
    if (L > 1e-6) {
        let dir = d / L;                // direction unitaire
        let stretch = L - params.rest;  // (L - L0)

        // Loi de Hooke : F = k * (L - L0) * dir
        (*force) = (*force) + params.k * stretch * dir;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let w = params.width;
    let h = params.height;

    // Coordonnées (x,y) dans la grille à partir de l’index i
    let x = i % w;
    let y = i / w;

    // -------------------------------
    // PINNING : on fixe 2 coins du haut
    // (x=0,y=0) et (x=w-1,y=0)
    // -------------------------------
    if ((y == 0u) && ((x == 0u) || (x == w - 1u))) {
        pos_out[i] = pos_in[i];
        vel_out[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Lecture position/vitesse d’entrée
    var p = pos_in[i].xyz;
    var v = vel_in[i].xyz;

    // Force totale initiale = gravité
    // F_g = m * g (sur Y)
    var F = vec3<f32>(0.0, params.mass * params.g, 0.0);

    // Voisins structuraux : gauche / droite / haut / bas
    if (x > 0u) {
        let j = idx_of(x - 1u, y);
        add_spring_force(p, pos_in[j].xyz, &F);
    }
    if (x + 1u < w) {
        let j = idx_of(x + 1u, y);
        add_spring_force(p, pos_in[j].xyz, &F);
    }
    if (y > 0u) {
        let j = idx_of(x, y - 1u);
        add_spring_force(p, pos_in[j].xyz, &F);
    }
    if (y + 1u < h) {
        let j = idx_of(x, y + 1u);
        add_spring_force(p, pos_in[j].xyz, &F);
    }

    // Intégration (Euler explicite)
    let a = F / params.mass;
    v = v + a * params.dt;

    // Amortissement simple : réduit l’oscillation
    v = v * params.damping;

    // Mise à jour de la position
    p = p + v * params.dt;

    // Écriture dans les buffers de sortie (double buffering)
    vel_out[i] = vec4<f32>(v, 0.0);
    pos_out[i] = vec4<f32>(p, 1.0);
}
