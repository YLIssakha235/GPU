struct Camera {
    mvp: mat4x4<f32>, // model-view-projection matrix
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VSIn {
    @location(0) position: vec4<f32>,
};

struct VSOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    out.clip = cam.mvp * in.position;
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.2, 0.2, 1.0);
}

