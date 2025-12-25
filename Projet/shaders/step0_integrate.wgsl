// Step 0 — Intégration : pos_out = pos_in + vel_in * dt
struct Params {
    dt: f32,
    n:  u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       pos_in  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       vel_in  : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> pos_out : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out : array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }

    let p = pos_in[i];
    let v = vel_in[i];

    pos_out[i] = vec4<f32>(p.xyz + v.xyz * params.dt, p.w);
    vel_out[i] = v; // passthrough
}
