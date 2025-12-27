import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.data_init import (
    make_grid_cloth,
    make_grid_line_indices,
    make_grid_indices,
    make_uv_sphere_wire,
    make_uv_sphere_triangles,   # ✅ AJOUT
)
from src.cloth_renderer_lit import ClothRendererLit
from src.sphere_renderer import SphereRenderer
from src.sphere_renderer_lit import SphereRendererLit  # ✅ AJOUT
from src.renderer import ClothRenderer
from src.camera import look_at, perspective


# ============================================================
# CONFIG SIMULATION (physique)
# ============================================================
G = -9.81

K_STRUCT = 60.0
K_SHEAR  = 80.0
K_BEND   = 300.0

DAMPING = 0.995
DT = 1 / 240
SUBSTEPS = 8
REST = 0.10
MASS = 0.1
WORKGROUP_SIZE = 64


# ============================================================
# CONFIG SPHÈRE / SOL (collision + friction)
# ============================================================
SPHERE_R = 0.8
MU = 0.6
EPS = 0.004
BOUNCE = 0.0
FLOOR_Y = 0.0


def main():
    # ------------------------------------------------------------
    # 1) GPU device
    # ------------------------------------------------------------
    device = get_default_device()
    print("✅ Device:", device)

    # ------------------------------------------------------------
    # 2) Sphère (source de vérité)
    # ------------------------------------------------------------
    sphere_cx, sphere_cz = 0.35, 0.0
    sphere_cy = 1.0
    sphere_r = SPHERE_R

    # ------------------------------------------------------------
    # 3) Tissu (CPU)
    # ------------------------------------------------------------
    W, H = 25, 25
    cloth_y0 = sphere_cy + sphere_r + 0.10

    positions_np, velocities_np = make_grid_cloth(
        W, H, REST,
        y0=cloth_y0,
        cx=sphere_cx,
        cz=sphere_cz,
    )
    velocities_np[:] = 0.0

    indices_np = make_grid_line_indices(W, H, diagonals=True)
    tri_indices_np = make_grid_indices(W, H)

    positions_np = np.asarray(positions_np, dtype=np.float32)
    velocities_np = np.asarray(velocities_np, dtype=np.float32)
    indices_np = np.asarray(indices_np, dtype=np.uint32)
    tri_indices_np = np.asarray(tri_indices_np, dtype=np.uint32)

    N = positions_np.shape[0]
    print("✅ Mesh:", positions_np.shape, "line:", indices_np.shape, "tri:", tri_indices_np.shape)

    # ------------------------------------------------------------
    # 4) Canvas / swapchain
    # ------------------------------------------------------------
    canvas = RenderCanvas(title="Cloth - Step5 (sphere + normals + surface)", size=(900, 700))
    context = canvas.get_context("wgpu")
    texture_format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=texture_format)

    # ------------------------------------------------------------
    # 5) Buffers GPU (ping-pong)
    # ------------------------------------------------------------
    pos_a = device.create_buffer_with_data(
        data=positions_np.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    pos_b = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    vel_a = device.create_buffer_with_data(
        data=velocities_np.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )
    vel_b = device.create_buffer(
        size=velocities_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
    )

    normal_buf = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )
    tri_idx_buf = device.create_buffer_with_data(
        data=tri_indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 6) Sphère : wireframe + triangles (surface) ✅
    # ------------------------------------------------------------
    # Wireframe
    sphere_pos_np, sphere_idx_np = make_uv_sphere_wire(stacks=16, slices=32)
    sphere_pos_buf = device.create_buffer_with_data(
        data=sphere_pos_np.tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    sphere_idx_buf = device.create_buffer_with_data(
        data=sphere_idx_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # Surface triangles ✅
    sphere_tri_pos_np, sphere_tri_idx_np = make_uv_sphere_triangles(stacks=16, slices=32)
    sphere_tri_pos_buf = device.create_buffer_with_data(
        data=sphere_tri_pos_np.tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    sphere_tri_idx_buf = device.create_buffer_with_data(
        data=sphere_tri_idx_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 7) Uniform buffers compute
    # ------------------------------------------------------------
    SPRINGS_PARAMS_SIZE = 48
    SPHERE_PARAMS_SIZE = 64
    NORMALS_PARAMS_SIZE = 16

    params_buf = device.create_buffer(size=SPRINGS_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    params_sphere_buf = device.create_buffer(size=SPHERE_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    params_normals_buf = device.create_buffer(size=NORMALS_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    # ------------------------------------------------------------
    # 8) Renderers
    # ------------------------------------------------------------
    renderer_lit = ClothRendererLit(canvas, device, tri_index_count=tri_indices_np.size)
    renderer_wire = ClothRenderer(canvas, device, index_count=indices_np.size, wireframe=True)

    sphere_renderer = SphereRenderer(canvas, device, index_count=sphere_idx_np.size)  # wire
    sphere_renderer_lit = SphereRendererLit(canvas, device, index_count=sphere_tri_idx_np.size)  # ✅ surface

    print("✅ Renderers OK")

    # ------------------------------------------------------------
    # 9) Camera
    # ------------------------------------------------------------
    aspect = 900 / 700
    model = np.eye(4, dtype=np.float32)

    target = (sphere_cx, sphere_cy, sphere_cz)
    eye = (sphere_cx, sphere_cy + 0.9, sphere_cz + 4.5)

    view = look_at(eye, target, (0.0, 1.0, 0.0))
    proj = perspective(70.0, aspect, 0.05, 50.0)

    mvp = proj @ view @ model
    mvp_bytes = mvp.T.astype(np.float32).tobytes()

    renderer_lit.set_mvp(mvp_bytes)
    renderer_wire.set_mvp(mvp_bytes)
    sphere_renderer.set_mvp(mvp_bytes)
    sphere_renderer_lit.set_mvp(mvp_bytes)  # ✅
    print("✅ Camera MVP set (centered on sphere)")

    # ------------------------------------------------------------
    # 10) Compute pipelines (identique à toi)
    # ------------------------------------------------------------
    springs_code = open("shaders/step2_structural_shear_bend.wgsl", "r", encoding="utf-8").read()
    springs_mod = device.create_shader_module(code=springs_code)

    springs_bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    springs_bg_a = device.create_bind_group(layout=springs_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_a, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": vel_a, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 2, "resource": {"buffer": pos_b, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 3, "resource": {"buffer": vel_b, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 4, "resource": {"buffer": params_buf, "offset": 0, "size": SPRINGS_PARAMS_SIZE}},
    ])
    springs_bg_b = device.create_bind_group(layout=springs_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_b, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": vel_b, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 2, "resource": {"buffer": pos_a, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 3, "resource": {"buffer": vel_a, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 4, "resource": {"buffer": params_buf, "offset": 0, "size": SPRINGS_PARAMS_SIZE}},
    ])

    springs_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[springs_bgl]),
        compute={"module": springs_mod, "entry_point": "main"},
    )

    sphere_code = open("shaders/step4_collision_friction.wgsl", "r", encoding="utf-8").read()
    sphere_mod = device.create_shader_module(code=sphere_code)

    sphere_bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    sphere_bg_a = device.create_bind_group(layout=sphere_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_a, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": vel_a, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 2, "resource": {"buffer": pos_b, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 3, "resource": {"buffer": vel_b, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 4, "resource": {"buffer": params_sphere_buf, "offset": 0, "size": SPHERE_PARAMS_SIZE}},
    ])
    sphere_bg_b = device.create_bind_group(layout=sphere_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_b, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": vel_b, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 2, "resource": {"buffer": pos_a, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 3, "resource": {"buffer": vel_a, "offset": 0, "size": velocities_np.nbytes}},
        {"binding": 4, "resource": {"buffer": params_sphere_buf, "offset": 0, "size": SPHERE_PARAMS_SIZE}},
    ])

    sphere_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[sphere_bgl]),
        compute={"module": sphere_mod, "entry_point": "main"},
    )

    normals_code = open("shaders/compute_normals_grid.wgsl", "r", encoding="utf-8").read()
    normals_mod = device.create_shader_module(code=normals_code)

    normals_bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    normals_bg_a = device.create_bind_group(layout=normals_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_a, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": normal_buf, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 2, "resource": {"buffer": params_normals_buf, "offset": 0, "size": NORMALS_PARAMS_SIZE}},
    ])
    normals_bg_b = device.create_bind_group(layout=normals_bgl, entries=[
        {"binding": 0, "resource": {"buffer": pos_b, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 1, "resource": {"buffer": normal_buf, "offset": 0, "size": positions_np.nbytes}},
        {"binding": 2, "resource": {"buffer": params_normals_buf, "offset": 0, "size": NORMALS_PARAMS_SIZE}},
    ])

    normals_pipeline = device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[normals_bgl]),
        compute={"module": normals_mod, "entry_point": "main"},
    )

    dispatch_x = (N + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    print(f"✅ dispatch_x={dispatch_x}")

    # ------------------------------------------------------------
    # 11) Boucle animation
    # ------------------------------------------------------------
    frame = 0
    ping = True
    dt_sub = np.float32(DT / SUBSTEPS)

    @canvas.request_draw
    def draw_frame():
        nonlocal frame, ping
        frame += 1

        cx, cy, cz = sphere_cx, sphere_cy, sphere_cz
        r = sphere_r

        for _ in range(SUBSTEPS):
            springs_params_bytes = b"".join([
                np.array([dt_sub, G, REST, MASS], dtype=np.float32).tobytes(),
                np.array([K_STRUCT, K_SHEAR, K_BEND, DAMPING], dtype=np.float32).tobytes(),
                np.array([W, H, N, 0], dtype=np.uint32).tobytes(),
            ])
            device.queue.write_buffer(params_buf, 0, springs_params_bytes)

            springs_bg = springs_bg_a if ping else springs_bg_b
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(springs_pipeline)
            cp.set_bind_group(0, springs_bg)
            cp.dispatch_workgroups(dispatch_x)
            cp.end()
            device.queue.submit([enc.finish()])
            ping = not ping

            sphere_params_bytes = b"".join([
                np.array([
                    dt_sub, cx, cy, cz,
                    r, BOUNCE, MU, EPS,
                    FLOOR_Y, 0.0, 0.0, 0.0
                ], dtype=np.float32).tobytes(),
                np.array([N, 0, 0, 0], dtype=np.uint32).tobytes(),
            ])
            device.queue.write_buffer(params_sphere_buf, 0, sphere_params_bytes)

            sphere_bg = sphere_bg_a if ping else sphere_bg_b
            enc = device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(sphere_pipeline)
            cp.set_bind_group(0, sphere_bg)
            cp.dispatch_workgroups(dispatch_x)
            cp.end()
            device.queue.submit([enc.finish()])
            ping = not ping

        current_pos = pos_a if ping else pos_b

        device.queue.write_buffer(params_normals_buf, 0, np.array([W, H, 0, 0], dtype=np.uint32).tobytes())
        normals_bg = normals_bg_a if (current_pos is pos_a) else normals_bg_b

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(normals_pipeline)
        cp.set_bind_group(0, normals_bg)
        cp.dispatch_workgroups(dispatch_x)
        cp.end()
        device.queue.submit([enc.finish()])

        tex = context.get_current_texture()
        view_tex = tex.create_view()
        enc = device.create_command_encoder()

        # sphère = physique
        sphere_renderer.set_sphere((cx, cy, cz), r)
        sphere_renderer_lit.set_sphere((cx, cy, cz), r)  # ✅

        # 1) tissu surface
        renderer_lit.encode(enc, view_tex, current_pos, normal_buf, tri_idx_buf, clear=True)

        # 2) sphère surface ✅
        sphere_renderer_lit.encode(enc, view_tex, sphere_tri_pos_buf, sphere_tri_idx_buf, clear=False)

        # 3) tissu wireframe
        renderer_wire.encode(enc, view_tex, current_pos, idx_buf, clear=False)

        # 4) sphère wireframe (optionnel)
        #sphere_renderer.encode(enc, view_tex, sphere_pos_buf, sphere_idx_buf, clear=False)

        device.queue.submit([enc.finish()])

        if frame % 60 == 0:
            print("frame", frame)

        canvas.request_draw()

    loop.run()


if __name__ == "__main__":
    main()
