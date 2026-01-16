// Étape 1 — Gravité
// v.y += g * dt
// pos += v * dt
//
// Lien avec le projet Cloth Simulation :
// - positions  : positions des sommets du tissu
// - velocities : vitesses des sommets
// - la gravité agit uniquement sur l’axe Y (vertical)

struct Params {
    dt: f32,   // pas de temps (Δt)
    g:  f32,   // accélération gravitationnelle
    n:  u32,   // nombre total de particules / sommets
    _pad: u32, // padding pour l’alignement mémoire
};

@group(0) @binding(0)
var<storage, read_write> positions : array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> velocities : array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {

    // Index global de la particule (un thread = une particule)
    let i = gid.x;

    // Sécurité : on évite de dépasser le nombre réel de particules
    if (i >= params.n) {
        return;
    }

    // Lecture des données actuelles
    var p = positions[i];
    var v = velocities[i];

    // Application de la gravité sur l’axe Y
    // v = v + a * dt
    v.y = v.y + params.g * params.dt;

    // Intégration d’Euler explicite :
    // p = p + v * dt
    p.x = p.x + v.x * params.dt;
    p.y = p.y + v.y * params.dt;
    p.z = p.z + v.z * params.dt;

    // Écriture des nouvelles valeurs dans les buffers GPU
    velocities[i] = v;
    positions[i]  = p;
}
