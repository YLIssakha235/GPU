// step3_collision_sphere_friction.wgsl
// Pass collision sphère + friction (Coulomb simple)
// Double buffering: pos_in/vel_in -> pos_out/vel_out

struct SphereParams {
    dt: f32,
    cx: f32,
    cy: f32,
    cz: f32,

    r: f32,
    bounce: f32,   // 0.0 tissu (pas rebond), 0.1 petit rebond...
    mu: f32,       // friction (0..1) typique: 0.05 à 0.4
    eps: f32,      // petit offset pour éviter re-collision numérique

    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,

    pad: vec4<f32>, // pour aligner à 64 bytes
};

@group(0) @binding(0) var<storage, read> pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : SphereParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    var p = pos_in[i].xyz;
    var v = vel_in[i].xyz;

    let c = vec3<f32>(params.cx, params.cy, params.cz);
    let r = params.r;

    // vecteur centre -> point
    let d = p - c;
    let dist = length(d);

    // Si à l'intérieur (ou trop proche), on corrige
    if (dist < r) {
        // normale (si dist trop petit, on prend une normale par défaut)
        let n = select(vec3<f32>(0.0, 1.0, 0.0), d / dist, dist > 1e-6);

        // 1) Projection sur la surface (avec eps)
        p = c + n * (r + params.eps);

        // 2) Décomposition vitesse en normal + tangentiel
        let vn = dot(v, n);
        var vt = v - vn * n;

        // On corrige seulement si on allait "vers l'intérieur" (vn < 0)
        if (vn < 0.0) {
            // 2a) bounce sur la normale
            // vn' = -bounce * vn  (vn est négatif)
            let vn_new = -params.bounce * vn;

            // 2b) friction Coulomb : on réduit la norme de vt
            // Impulsion normale ~ (1+bounce)*(-vn)
            let jn = (1.0 + params.bounce) * (-vn);

            let vt_len = length(vt);
            if (vt_len > 1e-6) {
                // quantité max qu'on peut enlever au tangent: mu * jn
                let drop = params.mu * jn;
                let vt_new_len = max(0.0, vt_len - drop);
                vt = vt * (vt_new_len / vt_len);
            }

            // Recomposition vitesse
            v = vt + vn_new * n;
        }
    }

    pos_out[i] = vec4<f32>(p, 1.0);
    vel_out[i] = vec4<f32>(v, 0.0);
}
