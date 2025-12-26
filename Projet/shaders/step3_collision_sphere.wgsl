// step3_collision_sphere.wgsl
// Étape 3 — Collision avec une sphère (projection + correction vitesse)
// Double buffering: pos_in/vel_in -> pos_out/vel_out

struct Params {
    dt: f32,
    sphere_cx: f32,
    sphere_cy: f32,
    sphere_cz: f32,

    sphere_r: f32,
    bounce: f32,      // 0 = pas de rebond (cloth), 0.1 petit rebond
    _pad0: f32,
    _pad1: f32,

    n: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
};

@group(0) @binding(0) var<storage, read>       pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    var p = pos_in[i].xyz;
    var v = vel_in[i].xyz;

    let c = vec3<f32>(params.sphere_cx, params.sphere_cy, params.sphere_cz);
    let r = params.sphere_r;

    let d = p - c;
    let dist = length(d);

    // Si point à l’intérieur de la sphère -> collision
    if (dist < r) {
        // normale (si dist ~ 0, on force une normale)
        var nrm = vec3<f32>(0.0, 1.0, 0.0);
        if (dist > 1e-6) {
            nrm = d / dist;
        }

        // Projection sur la surface
        p = c + nrm * r;

        // Correction vitesse: enlever composante vers l’intérieur
        // v' = v - (1 + bounce) * dot(v,n) * n  si dot(v,n) < 0
        let vn = dot(v, nrm);
        if (vn < 0.0) {
            v = v - (1.0 + params.bounce) * vn * nrm;
        }
    }

    pos_out[i] = vec4<f32>(p, 1.0);
    vel_out[i] = vec4<f32>(v, 0.0);
}
