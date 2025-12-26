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
K = 60.0            # raideur des ressorts (plus grand = tissu plus rigide)
DAMPING = 0.995     # amortissement (0.98..0.999). Trop petit -> tout s’écrase; trop grand -> ça explose
DT = 1 / 240        # pas de temps global
SUBSTEPS = 8        # sous-pas (stabilité collisions + ressorts)
REST = 0.10         # longueur au repos entre voisins
MASS = 1.0          # masse par particule (si ton shader l'utilise)
WORKGROUP_SIZE = 64 # doit correspondre à @workgroup_size(...) du shader


# ============================================================
# CONFIG SPHÈRE (collision + friction)
# ============================================================
SPHERE_R = 0.6      # rayon de la sphère
MU = 0.8         # friction (0 -> glisse, 0.2..0.4 -> accroche)
EPS = 0.004         # offset pour éviter de rester "collé" dans la sphère
BOUNCE = 0.0        # rebond (0 = aucun)


def main():
    # ------------------------------------------------------------
    # 1) GPU device
    # ------------------------------------------------------------
    device = get_default_device()
    print("✅ Device:", device)

    # ------------------------------------------------------------
    # 2) Définition "source de vérité" de la sphère
    #    IMPORTANT : on définit la sphère ici pour pouvoir
    #    placer le tissu AU-DESSUS dès l'initialisation.
    # ------------------------------------------------------------
    sphere_cx, sphere_cz = 0.35, 0.0   # léger décalage en X pour casser la symétrie
    sphere_cy = 1.0                   # hauteur de la sphère
    sphere_r = SPHERE_R

    # ------------------------------------------------------------
    # 3) Génération du tissu (CPU) - placé au-dessus de la sphère
    # ------------------------------------------------------------
    W, H = 10, 10

    # Hauteur initiale du tissu : centre sphère + rayon + marge
    cloth_y0 = sphere_cy + sphere_r + 0.10

    # On place le tissu centré (cx, cz) pareil que la sphère (ou légèrement décalé)
    positions_np, velocities_np = make_grid_cloth(
        W, H, REST,
        y0=cloth_y0,
        cx=sphere_cx,
        cz=sphere_cz,
    )

    # -------------------------------------------------
    # Petite poussée initiale pour casser la symétrie
    # (sinon aucune vitesse tangentielle → pas de glissement)
    # -------------------------------------------------
    velocities_np[:, 0] = 0.2   # poussée en X (essaie 0.1, 0.15, 0.3)


    # Index buffers : wireframe + triangles
    indices_np = make_grid_line_indices(W, H, diagonals=True)  # lignes
    tri_indices_np = make_grid_indices(W, H)                   # triangles

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

    # Normales (calculées au compute, utilisées au rendu lit)
    normal_buf = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    # Index buffers (wireframe + triangles)
    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )
    tri_idx_buf = device.create_buffer_with_data(
        data=tri_indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 6) Mesh sphère wireframe (rayon 1 au départ → on scale via set_sphere)
    # ------------------------------------------------------------
    sphere_pos_np, sphere_idx_np = make_uv_sphere_wire(stacks=16, slices=32)
    print("sphere pos range:", sphere_pos_np[:, 0].min(), sphere_pos_np[:, 0].max())

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
    # ------------------------------------------------------------
    SPRINGS_PARAMS_SIZE = 48
    params_buf = device.create_buffer(
        size=SPRINGS_PARAMS_SIZE,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    SPHERE_PARAMS_SIZE = 64
    params_sphere_buf = device.create_buffer(
        size=SPHERE_PARAMS_SIZE,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    NORMALS_PARAMS_SIZE = 16
    params_normals_buf = device.create_buffer(
        size=NORMALS_PARAMS_SIZE,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 8) Renderers
    # ------------------------------------------------------------
    renderer_lit = ClothRendererLit(canvas, device, tri_index_count=tri_indices_np.size)
    renderer_wire = ClothRenderer(canvas, device, index_count=indices_np.size, wireframe=True)
    sphere_renderer = SphereRenderer(canvas, device, index_count=sphere_idx_np.size)
    print("✅ Renderers OK")

    # ------------------------------------------------------------
    # 9) Camera (MVP)
    # ------------------------------------------------------------
    aspect = 900 / 700
    model = np.eye(4, dtype=np.float32)

    # Caméra un peu en retrait pour bien voir sphère + tissu
    view = look_at((2.5, 2.2, 2.5), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0))
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

    # ---------- B) SPHERE COLLISION ----------
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

    # ---------- C) NORMALS ----------
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

    # Nombre de workgroups (en X) pour traiter N particules
    dispatch_x = (N + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    print(f"✅ dispatch_x={dispatch_x}")

    # ------------------------------------------------------------
    # 11) Boucle animation
    # ------------------------------------------------------------
    frame = 0
    ping = True  # True => pos_a/vel_a sont l'entrée, False => pos_b/vel_b sont l'entrée
    dt_sub = np.float32(DT / SUBSTEPS)

    @canvas.request_draw
    def draw_frame():
        nonlocal frame, ping
        frame += 1

        # =========================================================
        # 0) SPHÈRE (source de vérité) : même valeurs pour physique + rendu
        # =========================================================
        cx, cy, cz = sphere_cx, sphere_cy, sphere_cz
        r = sphere_r

        # (si un jour tu veux animer : change cy/cx/cz ici)
        # t = frame * float(DT)
        # cy = sphere_cy + 0.2 * np.sin(t)

        # =========================================================
        # 1) PHYSIQUE (SUBSTEPS)
        # =========================================================
        for _ in range(SUBSTEPS):

            # ---------- PASS 1 : ressorts + gravité ----------
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

            # ---------- PASS 2 : collision sphère + friction ----------
            FLOOR_Y = 0.0  # sol à y=0 (à adapter si tu veux)

            sphere_params_bytes = b"".join([
                np.array([
                    dt_sub, cx, cy, cz,
                    r, BOUNCE, MU, EPS,
                    FLOOR_Y, 0.0, 0.0, 0.0   # padding float pour atteindre 12 floats
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

        # =========================================================
        # 2) Buffer position actuel (après ping-pong)
        # =========================================================
        current_pos = pos_a if ping else pos_b

        # =========================================================
        # 3) Calcul des normales (SUR LE MÊME buffer que le rendu)
        # =========================================================
        device.queue.write_buffer(
            params_normals_buf,
            0,
            np.array([W, H, 0, 0], dtype=np.uint32).tobytes()
        )
        normals_bg = normals_bg_a if (current_pos is pos_a) else normals_bg_b

        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(normals_pipeline)
        cp.set_bind_group(0, normals_bg)
        cp.dispatch_workgroups(dispatch_x)
        cp.end()
        device.queue.submit([enc.finish()])

        # =========================================================
        # 4) RENDU (une seule soumission)
        # =========================================================
        tex = context.get_current_texture()
        view = tex.create_view()
        enc = device.create_command_encoder()

        # IMPORTANT : sphère rendue = sphère physique
        sphere_renderer.set_sphere((cx, cy, cz), r)

        # 1) tissu surface (clear)
        renderer_lit.encode(enc, view, current_pos, normal_buf, tri_idx_buf, clear=True)

        # 2) tissu wireframe (load)
        renderer_wire.encode(enc, view, current_pos, idx_buf, clear=False)

        # 3) sphère wireframe (load)
        sphere_renderer.encode(enc, view, sphere_pos_buf, sphere_idx_buf, clear=False)

        device.queue.submit([enc.finish()])

        if frame % 60 == 0:
            print("frame", frame)

        canvas.request_draw()

    loop.run()


if __name__ == "__main__":
    main()
