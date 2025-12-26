// Collision sphère + friction (Coulomb) + collision sol (plan Y)
// => Ajout : friction STATIQUE (stick) + friction DYNAMIQUE (glisse)

struct SphereParams {
    // ---- 12 floats = 48 bytes ----
    dt: f32,
    cx: f32,
    cy: f32,
    cz: f32,

    r: f32,
    bounce: f32,
    mu: f32,
    eps: f32,

    floor_y: f32,
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

// ------------------------------------------------------------
// Frottement Coulomb avec "stick" (statique) + glisse (dynamique)
// jn_contact : "force normale" approx (>=0) même au repos
// ------------------------------------------------------------
fn apply_friction_static_dynamic(vt: vec3<f32>, jn_contact: f32, mu: f32) -> vec3<f32> {
    let vt_len = length(vt);
    if (vt_len < 1e-6) {
        return vec3<f32>(0.0);
    }

    let limit = mu * jn_contact;

    // FRICTION STATIQUE : si on a assez de marge, on colle
    // stick_k règle l'agressivité (2..8 typiquement)
    let stick_k = 8.0;
    if (vt_len * stick_k <= limit) {
        return vec3<f32>(0.0);
    }

    // FRICTION DYNAMIQUE : on réduit la norme tangentielle
    let vt_new_len = max(0.0, vt_len - limit);
    return vt * (vt_new_len / vt_len);
}

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
        let n = select(vec3<f32>(0.0, 1.0, 0.0), d / dist, dist > 1e-6);

        // Projection sur la surface
        let r_target = r + params.eps;
        let penetration = max(0.0, r_target - dist);
        p = c + n * r_target;

        // Décomposition vitesse
        let vn = dot(v, n);
        var vt = v - vn * n;

        // Corriger la composante normale (empêche d'entrer)
        var vn_corr = vn;
        if (vn < 0.0) {
            vn_corr = -params.bounce * vn;  // bounce=0 => vn_corr=0
        }

        // Contact normal (même au repos)
        let jn_impact = max(0.0, (1.0 + params.bounce) * (-vn));
        let jn_penetration = penetration / max(params.dt, 1e-6);
        let jn_contact = max(jn_impact, jn_penetration);

        // Frottement (statique+dynamique) sur la tangentielle
        vt = apply_friction_static_dynamic(vt, jn_contact, params.mu);

        // Recomposition propre
        v = vt + vn_corr * n;
    }


    // =========================================================
    // B) Collision SOL (plan y = floor_y) + friction
    // =========================================================
    if (p.y < params.floor_y) {
        p.y = params.floor_y + params.eps;

        let vy_in = v.y;
        if (vy_in < 0.0) {
            // bounce
            v.y = -params.bounce * vy_in;

            // tangent sol = (x,z)
            let vt3 = vec3<f32>(v.x, 0.0, v.z);

            // contact normal (même idée)
            let penetration = (params.floor_y + params.eps) - p.y; // ~0 après projection
            let jn_impact = max(0.0, (1.0 + params.bounce) * (-vy_in));
            let jn_contact = jn_impact; // sol: impact suffit généralement

            let vt3_new = apply_friction_static_dynamic(vt3, jn_contact, params.mu);
            v.x = vt3_new.x;
            v.z = vt3_new.z;
        }
    }

    pos_out[i] = vec4<f32>(p, 1.0);
    vel_out[i] = vec4<f32>(v, 0.0);
}
