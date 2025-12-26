import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.data_init import (
    make_grid_cloth,
    make_grid_line_indices,
    make_grid_indices,
    make_uv_sphere_wire,
)
from src.cloth_renderer_lit import ClothRendererLit
from src.sphere_renderer import SphereRenderer
from src.renderer import ClothRenderer
from src.camera import look_at, perspective


# ============================================================
# CONFIG SIMULATION (physique)
# ============================================================
G = -9.81           # gravité (m/s²)
K = 60.0            # raideur ressorts (plus grand => tissu plus rigide)
DAMPING = 0.995     # amortissement global (0.98..0.999)
DT = 1 / 240        # pas de temps global
SUBSTEPS = 8        # sous-pas (stabilité)
REST = 0.10         # longueur au repos entre voisins
MASS = 0.1         # masse (si utilisée dans ton shader)
WORKGROUP_SIZE = 64 # doit matcher @workgroup_size(64)


# ============================================================
# CONFIG SPHÈRE / SOL (collision + friction)
# ============================================================
SPHERE_R = 0.8      # rayon sphère
MU = 0.6           # friction (0=glisse, ~0.2..0.6 réaliste)
EPS = 0.004         # petit offset anti-collage / anti-penetration
BOUNCE = 0.0        # rebond (0 = aucun)
FLOOR_Y = 0.0       # hauteur du sol (plan y=floor_y)


def main():
    # ------------------------------------------------------------
    # 1) GPU device
    # ------------------------------------------------------------
    device = get_default_device()
    print("✅ Device:", device)

    # ------------------------------------------------------------
    # 2) "Source de vérité" de la sphère (PHYSIQUE + RENDU)
    #    -> On définit ici les valeurs officielles.
    # ------------------------------------------------------------
    sphere_cx, sphere_cz = 0.35, 0.0   # petit décalage X pour casser la symétrie
    sphere_cy = 1.0                   # hauteur centre sphère
    sphere_r = SPHERE_R               # rayon sphère

    # ------------------------------------------------------------
    # 3) Tissu (CPU) : placé AU-DESSUS de la sphère
    # ------------------------------------------------------------
    W, H = 20, 20  # résolution du tissu (points, pas triangles)
    cloth_y0 = sphere_cy + sphere_r + 0.10  # centre sphère + rayon + marge

    positions_np, velocities_np = make_grid_cloth(
        W, H, REST,
        y0=cloth_y0,
        cx=sphere_cx,
        cz=sphere_cz,
    )

    # IMPORTANT :
    # - Pour “un tissu qui se pose tranquille”, évite une grosse vitesse initiale.
    # - Si tu veux juste casser la symétrie, mets un TRÈS petit bruit.
    #rng = np.random.default_rng(0)
    #velocities_np[:, 0] += (rng.uniform(-1.0, 1.0, size=(velocities_np.shape[0],)) * 0.01).astype(np.float32)
    # aucune vitesse initiale
    velocities_np[:] = 0.0


    # Index buffers (wire + triangles)
    indices_np = make_grid_line_indices(W, H, diagonals=True)  # lignes (wireframe)
    tri_indices_np = make_grid_indices(W, H)                   # triangles (surface)

    # Conversion explicite (important)
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
    # 5) Buffers GPU (ping-pong positions + vitesses)
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

    # Normales (compute -> rendu lit)
    normal_buf = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    # Index buffers
    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )
    tri_idx_buf = device.create_buffer_with_data(
        data=tri_indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 6) Mesh sphère wireframe (rayon 1 -> on scale dans SphereRenderer.set_sphere)
    # ------------------------------------------------------------
    sphere_pos_np, sphere_idx_np = make_uv_sphere_wire(stacks=16, slices=32)

    sphere_pos_buf = device.create_buffer_with_data(
        data=sphere_pos_np.tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    sphere_idx_buf = device.create_buffer_with_data(
        data=sphere_idx_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 7) Uniform buffers (params compute)
    #    ATTENTION: tailles doivent matcher les structs WGSL
    # ------------------------------------------------------------
    SPRINGS_PARAMS_SIZE = 48   # 8 floats (32) + 4 u32 (16) = 48
    SPHERE_PARAMS_SIZE = 64    # 12 floats (48) + 4 u32 (16) = 64  (ton shader step4)
    NORMALS_PARAMS_SIZE = 16   # 4 u32 = 16

    params_buf = device.create_buffer(size=SPRINGS_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    params_sphere_buf = device.create_buffer(size=SPHERE_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    params_normals_buf = device.create_buffer(size=NORMALS_PARAMS_SIZE, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

    # ------------------------------------------------------------
    # 8) Renderers
    # ------------------------------------------------------------
    renderer_lit = ClothRendererLit(canvas, device, tri_index_count=tri_indices_np.size)
    renderer_wire = ClothRenderer(canvas, device, index_count=indices_np.size, wireframe=True)
    sphere_renderer = SphereRenderer(canvas, device, index_count=sphere_idx_np.size)
    print("✅ Renderers OK")

    # ------------------------------------------------------------
    # 9) Camera (MVP)
    #    "en face" = caméra alignée Z, sans décalage X.
    # ------------------------------------------------------------
    aspect = 900 / 700
    model = np.eye(4, dtype=np.float32)

    # Caméra "en face"
    # - eye : position caméra
    # - target : point regardé
    view = look_at((0.0, 1.8, 3.5), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    proj = perspective(60.0, aspect, 0.1, 100.0)

    mvp = proj @ view @ model
    mvp_bytes = mvp.T.astype(np.float32).tobytes()

    renderer_lit.set_mvp(mvp_bytes)
    renderer_wire.set_mvp(mvp_bytes)
    sphere_renderer.set_mvp(mvp_bytes)
    print("✅ Camera MVP set")

    # ------------------------------------------------------------
    # 10) Compute pipelines
    # ------------------------------------------------------------
    # ---------- A) SPRINGS ----------
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

    # ---------- B) COLLISION SPHÈRE + SOL (ton shader step4_collision_friction.wgsl) ----------
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

    # ---------- C) NORMALES ----------
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

    # Workgroups nécessaires
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

        # =========================
        # Sphère (valeurs officielles)
        # =========================
        cx, cy, cz = sphere_cx, sphere_cy, sphere_cz
        r = sphere_r

        # =========================================================
        # 1) PHYSIQUE (SUBSTEPS)
        # =========================================================
        for _ in range(SUBSTEPS):

            # ---------- PASS 1 : ressorts + gravité ----------
            springs_params_bytes = b"".join([
                # 8 floats
                np.array([dt_sub, G, K, REST, MASS, DAMPING, 0.0, 0.0], dtype=np.float32).tobytes(),
                # 4 u32
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

            # ---------- PASS 2 : collision sphère + friction + sol ----------
            # IMPORTANT: doit matcher EXACTEMENT ton struct WGSL (12 floats + 4 u32)
            sphere_params_bytes = b"".join([
                # 12 floats
                np.array([
                    dt_sub, cx, cy, cz,
                    r, BOUNCE, MU, EPS,
                    FLOOR_Y, 0.0, 0.0, 0.0
                ], dtype=np.float32).tobytes(),
                # 4 u32
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

        # =========================================================
        # 2) Buffer de position courant (après ping-pong)
        # =========================================================
        current_pos = pos_a if ping else pos_b

        # =========================================================
        # 3) Normales (sur le même buffer que celui rendu)
        # =========================================================
        device.queue.write_buffer(params_normals_buf, 0, np.array([W, H, 0, 0], dtype=np.uint32).tobytes())
        normals_bg = normals_bg_a if (current_pos is pos_a) else normals_bg_b

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(normals_pipeline)
        cp.set_bind_group(0, normals_bg)
        cp.dispatch_workgroups(dispatch_x)
        cp.end()
        device.queue.submit([enc.finish()])

        # =========================================================
        # 4) Rendu (une seule submit)
        # =========================================================
        tex = context.get_current_texture()
        view_tex = tex.create_view()
        enc = device.create_command_encoder()

        # Sphère rendue = sphère physique
        sphere_renderer.set_sphere((cx, cy, cz), r)

        # 1) tissu surface (clear)
        renderer_lit.encode(enc, view_tex, current_pos, normal_buf, tri_idx_buf, clear=True)

        # 2) tissu wireframe (load)
        renderer_wire.encode(enc, view_tex, current_pos, idx_buf, clear=False)

        # 3) sphère wireframe (load)
        sphere_renderer.encode(enc, view_tex, sphere_pos_buf, sphere_idx_buf, clear=False)

        device.queue.submit([enc.finish()])

        if frame % 60 == 0:
            print("frame", frame)

        canvas.request_draw()

    loop.run()


if __name__ == "__main__":
    main()
