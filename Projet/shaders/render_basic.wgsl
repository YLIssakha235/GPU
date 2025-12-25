struct VSIn {
    @location(0) position: vec4<f32>,
};

struct VSOut {
    @builtin(position) clip: vec4<f32>,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
    var out: VSOut;
    // On prend x,z (ou x,y selon ton make_grid_cloth) et on scale un peu
    out.clip = vec4<f32>(in.position.x * 0.7, in.position.z * 0.7, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.2, 0.2, 1.0);
}
