struct Params {
    w: u32,
    h: u32,
    _0: u32,
    _1: u32,
};


@group(0) @binding(0) var<storage, read> pos : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> nrm : array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params : Params;

fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
  let l = length(v);
  if (l < 1e-8) { return vec3<f32>(0.0, 1.0, 0.0); }
  return v / l;
}

fn idx(i: i32, j: i32, w: i32) -> u32 { return u32(j*w + i); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x;
  let N = params.w * params.h;
  if (id >= N) { return; }

  let w = i32(params.w);
  let h = i32(params.h);

  let i = i32(id % params.w);
  let j = i32(id / params.w);

  let il = max(i - 1, 0);
  let ir = min(i + 1, w - 1);
  let jd = max(j - 1, 0);
  let ju = min(j + 1, h - 1);

  let pL = pos[idx(il, j, w)].xyz;
  let pR = pos[idx(ir, j, w)].xyz;
  let pD = pos[idx(i, jd, w)].xyz;
  let pU = pos[idx(i, ju, w)].xyz;

  let dx = pR - pL;
  let dz = pU - pD;

  let n = safe_normalize(cross(dz, dx));
  nrm[id] = vec4<f32>(n, 0.0);
}
