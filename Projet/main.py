import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.data_init import make_grid_cloth, make_grid_line_indices
from src.renderer import ClothRenderer
from src.camera import look_at, perspective


# ============================================================
# CONFIG SIMULATION (CPU)
# ============================================================
G = -9.81
K = 60.0
DAMPING = 0.985
DT = 1 / 240
SUBSTEPS = 4
REST = 0.10
MASS = 1.0
WORKGROUP_SIZE = 64  # doit correspondre au shader

# ============================================================
# CONFIG SPHERE COLLISION + FRICTION
# ============================================================
SPHERE_CX = 0.0
SPHERE_CY = 0.2
SPHERE_CZ = 0.0
SPHERE_R  = 0.9

MU = 0.15      # friction (0.05..0.4)
EPS = 1e-3     # petit offset (1e-4..1e-2)
BOUNCE = 0.0   # tissu = 0


def main():
    device = get_default_device()
    print("✅ Device:", device)

    # ============================================================
    # 1) MESH CPU : positions + velocities + indices (wireframe)
    # ============================================================
    W, H = 20, 20
    positions_np, velocities_np = make_grid_cloth(W, H, REST)
    indices_np = make_grid_line_indices(W, H, diagonals=True)

    positions_np  = np.asarray(positions_np, dtype=np.float32)     # (N,4)
    velocities_np = np.asarray(velocities_np, dtype=np.float32)    # (N,4)
    indices_np    = np.asarray(indices_np, dtype=np.uint32)        # (M,)

    N = positions_np.shape[0]
    print("pos x range:", positions_np[:, 0].min(), positions_np[:, 0].max())
    print("pos z range:", positions_np[:, 2].min(), positions_np[:, 2].max())
    print("✅ Mesh:", positions_np.shape, "line indices:", indices_np.shape)

    # ============================================================
    # 2) CANVAS
    # ============================================================
    canvas = RenderCanvas(title="Cloth - Step4 (springs + sphere + friction)", size=(900, 700))

    # ============================================================
    # 3) BUFFERS GPU (PING-PONG)
    # ============================================================
    pos_a = device.create_buffer_with_data(
        data=positions_np.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
    )
    pos_b = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
    )

    vel_a = device.create_buffer_with_data(
        data=velocities_np.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )
    vel_b = device.create_buffer(
        size=velocities_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
    )

    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
    )

    # ============================================================
    # 3B) UNIFORMS
    # ============================================================
    # Springs (2C) Params = 48 bytes (8 floats + 4 u32)
    SPRINGS_PARAMS_SIZE = 48
    params_buf = device.create_buffer(
        size=SPRINGS_PARAMS_SIZE,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    # Sphere+friction shader (ton step4_collision_friction) attend 64 bytes
    # (selon ton struct SphereParams avec pad vec4)
    SPHERE_PARAMS_SIZE = 64
    params_sphere_buf = device.create_buffer(
        size=SPHERE_PARAMS_SIZE,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
    )

    # ============================================================
    # 4) RENDERER (wireframe + camera)
    # ============================================================
    renderer = ClothRenderer(canvas, device, index_count=indices_np.size, wireframe=True)
    print("✅ Renderer OK")

    # ============================================================
    # 5) CAMERA MVP
    # ============================================================
    w_px, h_px = 900, 700
    aspect = w_px / h_px

    model = np.eye(4, dtype=np.float32)
    eye    = (0.0, 2.0, 2.8)
    target = (0.0, 1.0, 0.0)
    up     = (0.0, 1.0, 0.0)

    view = look_at(eye, target, up)
    proj = perspective(45.0, aspect, 0.1, 100.0)
    mvp = proj @ view @ model
    renderer.set_mvp(mvp.T.astype(np.float32).tobytes())
    print("✅ Camera set.")

    # ============================================================
    # 6) COMPUTE PASS 1 : SPRINGS (2C)
    # ============================================================
    with open("shaders/step2_structural_shear_bend.wgsl", "r", encoding="utf-8") as f:
        springs_code = f.read()
    springs_mod = device.create_shader_module(code=springs_code)

    springs_bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])

    # Ping-pong bindgroups
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

    springs_pl = device.create_pipeline_layout(bind_group_layouts=[springs_bgl])
    springs_pipeline = device.create_compute_pipeline(
        layout=springs_pl,
        compute={"module": springs_mod, "entry_point": "main"},
    )

    dispatch_x = (N + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    print(f"✅ Springs compute ready (dispatch_x={dispatch_x})")

    # ============================================================
    # 6B) COMPUTE PASS 2 : SPHERE COLLISION + FRICTION
    # ============================================================
    with open("shaders/step4_collision_friction.wgsl", "r", encoding="utf-8") as f:
        sphere_code = f.read()
    sphere_mod = device.create_shader_module(code=sphere_code)

    sphere_bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
         "buffer": {"type": wgpu.BufferBindingType.uniform}},
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

    sphere_pl = device.create_pipeline_layout(bind_group_layouts=[sphere_bgl])
    sphere_pipeline = device.create_compute_pipeline(
        layout=sphere_pl,
        compute={"module": sphere_mod, "entry_point": "main"},
    )
    print("✅ Sphere+friction compute ready")

    # ============================================================
    # 7) LOOP : springs + sphere in SUBSTEPS, then render
    # ============================================================
    frame = 0
    ping = True
    dt_sub = np.float32(DT / SUBSTEPS)

    @canvas.request_draw
    def draw_frame():
        nonlocal frame, ping
        frame += 1

        for _ in range(SUBSTEPS):
            # ---- PASS 1: SPRINGS (2C) ----
            springs_params_bytes = b"".join([
                np.array([dt_sub, G, K, REST, MASS, DAMPING, 0.0, 0.0], dtype=np.float32).tobytes(),
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

            # ---- PASS 2: SPHERE COLLISION + FRICTION ----
            sphere_params_bytes = b"".join([
                np.array([dt_sub, SPHERE_CX, SPHERE_CY, SPHERE_CZ,
                          SPHERE_R, BOUNCE, MU, EPS], dtype=np.float32).tobytes(),
                np.array([N, 0, 0, 0], dtype=np.uint32).tobytes(),
                np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32).tobytes(),  # pad vec4 -> 64
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

        # Render the last written buffer
        current_pos = pos_a if ping else pos_b
        renderer.draw(current_pos, idx_buf)

        if frame % 60 == 0:
            print("frame", frame)

        canvas.request_draw()

    # démarre le rendu (configure la surface)
    canvas.request_draw()
    loop.run()


if __name__ == "__main__":
    main()
