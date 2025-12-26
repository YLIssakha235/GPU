// step3_collision_sphere_friction.wgsl
// Collision sphère + friction (Coulomb simple) + collision sol (plan Y)
// Double buffering: pos_in/vel_in -> pos_out/vel_out

struct SphereParams {
    // ---- 12 floats = 48 bytes ----
    dt: f32,
    cx: f32,
    cy: f32,
    cz: f32,

    r: f32,
    bounce: f32,   // 0.0 = pas rebond, 0.1 = léger rebond
    mu: f32,       // friction (0..1) typique: 0.05 à 0.4
    eps: f32,      // petit offset anti "re-collision"

    floor_y: f32,  // hauteur du sol (plan horizontal)
    _pad_f0: f32,
    _pad_f1: f32,
    _pad_f2: f32,

    // ---- 4 u32 = 16 bytes ----
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

    // =========================================================
    // A) Collision SPHÈRE + friction
    // =========================================================
    let d = p - c;
    let dist = length(d);

    if (dist < r) {
        // normale (si dist trop petit, normale par défaut)
        let n = select(vec3<f32>(0.0, 1.0, 0.0), d / dist, dist > 1e-6);

        // 1) Projection sur la surface (avec eps)
        p = c + n * (r + params.eps);

        // 2) Décomposition vitesse en normal + tangentiel
        let vn = dot(v, n);
        var vt = v - vn * n;

        // On corrige seulement si on allait vers l'intérieur (vn < 0)
        if (vn < 0.0) {
            // rebond sur la normale
            let vn_new = -params.bounce * vn;

            // friction Coulomb : on réduit la norme de vt
            let jn = (1.0 + params.bounce) * (-vn); // impulsion normale approx

            let vt_len = length(vt);
            if (vt_len > 1e-6) {
                let drop = params.mu * jn;
                let vt_new_len = max(0.0, vt_len - drop);
                vt = vt * (vt_new_len / vt_len);
            }

            v = vt + vn_new * n;
        }
    }

    // =========================================================
    // B) Collision SOL (plan y = floor_y) + friction
    //    -> empêche le tissu de "disparaître" (il tombe dans le vide sinon)
    // =========================================================
    if (p.y < params.floor_y) {
        // on remonte juste au-dessus du sol
        p.y = params.floor_y + params.eps;

        // si on descendait, on coupe / rebondit la vitesse verticale
        let vy_in = v.y;
        if (vy_in < 0.0) {
            v.y = -params.bounce * vy_in; // bounce=0 -> v.y = 0

            // friction sur le sol: on réduit vx/vz (tangent)
            let vt = vec2<f32>(v.x, v.z);
            let vt_len = length(vt);

            if (vt_len > 1e-6) {
                // impulsion normale approx liée à la vitesse verticale d'impact
                let jn = (1.0 + params.bounce) * (-vy_in);
                let drop = params.mu * jn;

                let vt_new_len = max(0.0, vt_len - drop);
                let vt2 = vt * (vt_new_len / vt_len);

                v.x = vt2.x;
                v.z = vt2.y;
            }
        }
    }

    pos_out[i] = vec4<f32>(p, 1.0);
    vel_out[i] = vec4<f32>(v, 0.0);
}
