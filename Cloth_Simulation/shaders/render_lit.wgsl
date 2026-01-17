struct Camera {
  mvp: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VSIn {
  @location(0) position: vec4<f32>,
  @location(1) normal: vec4<f32>,   // normal.xyz
};

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) n: vec3<f32>,
};

@vertex
fn vs_main(v: VSIn) -> VSOut {
  var o: VSOut;
  o.clip = cam.mvp * v.position;
  o.n = normalize(v.normal.xyz);
  return o;
}

@fragment
fn fs_main(i: VSOut) -> @location(0) vec4<f32> {
  // direction lumière
  let L = normalize(vec3<f32>(0.3, 1.0, 0.4));
  //let L = normalize(vec3<f32>(-1.0, 0.2, 0.0));

  let ndotl = max(0.0, dot(i.n, L));

  // couleur tissu 
  let base = vec3<f32>(0.1, 0.6, 1.0);
  let ambient = 0.25;
  let col = base * (ambient + (1.0 - ambient) * ndotl);

  return vec4<f32>(col, 1.0);
}
