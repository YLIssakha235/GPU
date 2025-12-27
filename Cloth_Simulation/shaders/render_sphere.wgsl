struct Camera { mvp: mat4x4<f32> };
@group(0) @binding(0) var<uniform> cam: Camera;

// 7 * 16 = 112 bytes
struct SphereU {
  data: array<vec4<f32>, 7>,
};
@group(1) @binding(0) var<uniform> sph: SphereU;

struct VSIn { @location(0) position: vec4<f32> };
struct VSOut { @builtin(position) clip: vec4<f32> };

@vertex
fn vs_main(v: VSIn) -> VSOut {
  var o: VSOut;

  let center = sph.data[0].xyz;
  let radius = sph.data[1].x;

  let world = v.position.xyz * radius + center;
  o.clip = cam.mvp * vec4<f32>(world, 1.0);
  return o;
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(1.0, 0.8, 0.2, 1.0);
}
