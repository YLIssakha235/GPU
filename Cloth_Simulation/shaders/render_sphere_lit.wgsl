struct Camera {
    mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;


struct SphereU {
    v0: vec4<f32>, // (cx, cy, cz, 0)
    v1: vec4<f32>, // (r, 0, 0, 0)
    v2: vec4<f32>,
    v3: vec4<f32>,
    v4: vec4<f32>,
    v5: vec4<f32>,
    v6: vec4<f32>,
};
@group(1) @binding(0) var<uniform> sph: SphereU;

struct VSIn {
    @location(0) pos: vec4<f32>, // unit sphere vertex (x,y,z,1)
};

struct VSOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let c = sph.v0.xyz;
    let r = sph.v1.x;

    let unit = input.pos.xyz; // point sur sphère unité
    let world = c + r * unit; 

    var out: VSOut;
    out.normal = normalize(unit);
    out.clip = cam.mvp * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.3, 0.9, 0.2));
    let ndl = max(0.0, dot(input.normal, light_dir));

    let base = vec3<f32>(0.1, 0.1, 0.28);
    let col = base * (0.3 + 0.7 * ndl);

    return vec4<f32>(col, 1.0);
}

