// Étape 0 — Intégration des positions avec vitesses constantes
// Objectif : pos[i] = pos[i] + vel[i] * dt
//
// Lien avec le projet Cloth Simulation :
// - positions : positions des sommets du tissu
// - velocities : vitesses des sommets
// - dt : pas de temps de la simulation

struct Params {
    dt: f32,   // pas de temps (Δt)
    n:  u32,   // nombre total de points / sommets
    _pad0: u32, // padding mémoire (alignement)
    _pad1: u32, // padding mémoire (alignement)
};

@group(0) @binding(0) var<storage, read_write> positions : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities : array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {

    // Index global du point traité (un thread = un sommet)
    let i = gid.x;

    // Vérification de sécurité pour éviter tout dépassement de buffer
    if (i >= params.n) {
        return;
    }

    // Lecture de la position et de la vitesse actuelles
    let p = positions[i];
    let v = velocities[i];

    // Intégration temporelle (Euler explicite)
    // p(t+dt) = p(t) + v(t) * dt
    positions[i] = vec4<f32>(p.xyz + v.xyz * params.dt, p.w);
}
