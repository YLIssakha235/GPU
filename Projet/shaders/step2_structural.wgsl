// Étape 2A — Ressorts structuraux (voisins directs) + gravité
// Double buffering : pos_in/vel_in → pos_out/vel_out

// --------------------
// Uniforms (alignement propre)
// --------------------
struct Params {
    dt      : f32,
    g       : f32,
    k       : f32,
    rest    : f32,

    mass    : f32,
    damping : f32,
    _pad0   : f32,
    _pad1   : f32,

    width   : u32,
    height  : u32,
    n       : u32,
    _pad2   : u32,
};

// --------------------
// Buffers
// --------------------
@group(0) @binding(0) var<storage, read>       pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : Params;

// (x,y) → index linéaire
fn idx_of(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Ajout force ressort Hooke entre p (courant) et q (voisin) : F += k*(L-L0)*dir
fn add_spring_force(p: vec3<f32>, q: vec3<f32>, F: ptr<function, vec3<f32>>) {
    let d = q - p;
    let L = length(d);

    if (L > 1e-6) {
        let dir = d / L;
        let stretch = L - params.rest;
        (*F) = (*F) + params.k * stretch * dir;
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let w = params.width;
    let h = params.height;

    let x = i % w;
    let y = i / w;

    // -------------------------------
    // PINNING : 2 coins du haut
    // -------------------------------
    if ((y == 0u) && ((x == 0u) || (x == w - 1u))) {
        pos_out[i] = pos_in[i];
        vel_out[i] = vec4<f32>(0.0, 0.0, 0.0, vel_in[i].w);
        return;
    }

    // Lecture entrée
    let pin = pos_in[i];
    let vin = vel_in[i];

    var p = pin.xyz;
    var v = vin.xyz;

    // Force totale : gravité
    var F = vec3<f32>(0.0, params.mass * params.g, 0.0);

    // Voisins structuraux (gauche/droite/haut/bas)
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

    // Intégration Euler
    let a = F / params.mass;
    v = v + a * params.dt;

    // Amortissement
    v = v * params.damping;

    // Position
    p = p + v * params.dt;

    // Écriture sortie (on conserve w)
    vel_out[i] = vec4<f32>(v, vin.w);
    pos_out[i] = vec4<f32>(p, pin.w);
}
