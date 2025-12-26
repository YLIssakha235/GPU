struct Params {
    dt: f32,
    g:  f32,
    n:  u32,
    _pad: u32,
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

    var p = pos_in[i];
    var v = vel_in[i];

    // gravity
    v.y = v.y + params.g * params.dt;

    // integrate (WGSL: pas d'assignation sur swizzle p.xyz)
    p.x = p.x + v.x * params.dt;
    p.y = p.y + v.y * params.dt;
    p.z = p.z + v.z * params.dt;

    vel_out[i] = v;
    pos_out[i] = p;
}
