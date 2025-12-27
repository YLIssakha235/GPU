// Étape 2C — Ressorts structural + shear + bend + gravité
// Double buffering : pos_in/vel_in → pos_out/vel_out

struct Params {
    // bloc 0 (4 x f32)
    dt: f32,
    g: f32,
    rest: f32,
    mass: f32,

    // bloc 1 (4 x f32)
    k_struct: f32,
    k_shear: f32,
    k_bend: f32,
    damping: f32,

    // bloc 2 (4 x u32)
    width: u32,
    height: u32,
    n: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>  pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : Params;

fn idx_of(x: u32, y: u32) -> u32 {
    return y * params.width + x;
}

// Force ressort avec longueur au repos L0 donnée + raideur k donnée
fn add_spring_force_L0(
    p: vec3<f32>,
    q: vec3<f32>,
    L0: f32,
    k: f32,
    force: ptr<function, vec3<f32>>
) {
    let d = q - p;
    let L = length(d);
    if (L > 1e-6) {
        let dir = d / L;
        let stretch = L - L0;
        (*force) = (*force) + k * stretch * dir;
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

    var p = pos_in[i].xyz;
    var v = vel_in[i].xyz;

    // gravité
    var F = vec3<f32>(0.0, params.mass * params.g, 0.0);

    // longueurs au repos
    let L0_struct = params.rest;
    let L0_shear  = params.rest * 1.41421356237; // sqrt(2)
    let L0_bend   = params.rest * 2.0;

    // ---------- Structural (4 voisins) ----------
    if (x > 0u) {
        let j = idx_of(x - 1u, y);
        add_spring_force_L0(p, pos_in[j].xyz, L0_struct, params.k_struct, &F);
    }
    if (x + 1u < w) {
        let j = idx_of(x + 1u, y);
        add_spring_force_L0(p, pos_in[j].xyz, L0_struct, params.k_struct, &F);
    }
    if (y > 0u) {
        let j = idx_of(x, y - 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_struct, params.k_struct, &F);
    }
    if (y + 1u < h) {
        let j = idx_of(x, y + 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_struct, params.k_struct, &F);
    }

    // ---------- Shear (4 diagonales) ----------
    if (x > 0u && y > 0u) {
        let j = idx_of(x - 1u, y - 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_shear, params.k_shear, &F);
    }
    if (x + 1u < w && y > 0u) {
        let j = idx_of(x + 1u, y - 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_shear, params.k_shear, &F);
    }
    if (x > 0u && y + 1u < h) {
        let j = idx_of(x - 1u, y + 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_shear, params.k_shear, &F);
    }
    if (x + 1u < w && y + 1u < h) {
        let j = idx_of(x + 1u, y + 1u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_shear, params.k_shear, &F);
    }

    // ---------- Bend (distance 2) ----------
    if (x >= 2u) {
        let j = idx_of(x - 2u, y);
        add_spring_force_L0(p, pos_in[j].xyz, L0_bend, params.k_bend, &F);
    }
    if (x + 2u < w) {
        let j = idx_of(x + 2u, y);
        add_spring_force_L0(p, pos_in[j].xyz, L0_bend, params.k_bend, &F);
    }
    if (y >= 2u) {
        let j = idx_of(x, y - 2u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_bend, params.k_bend, &F);
    }
    if (y + 2u < h) {
        let j = idx_of(x, y + 2u);
        add_spring_force_L0(p, pos_in[j].xyz, L0_bend, params.k_bend, &F);
    }

    // intégration
    let a = F / params.mass;
    v = v + a * params.dt;
    v = v * params.damping;
    p = p + v * params.dt;

    vel_out[i] = vec4<f32>(v, 0.0);
    pos_out[i] = vec4<f32>(p, 1.0);
}
