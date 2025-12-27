import numpy as np
import inspect
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.data_init import (
    make_grid_cloth,
    make_grid_line_indices,
    make_grid_indices,
    make_uv_sphere_wire,
    make_uv_sphere_triangles,
)
from src.cloth_renderer_lit import ClothRendererLit
from src.sphere_renderer import SphereRenderer
from src.sphere_renderer_lit import SphereRendererLit
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
    W, H = 12, 12
    cloth_y0 = sphere_cy + sphere_r + 0.10

    positions_np, velocities_np = make_grid_cloth(
        W, H, REST,
        y0=cloth_y0,
        cx=sphere_cx,
        cz=sphere_cz,
    )
    velocities_np[:] = 0.0

    # On garde une copie CPU pour "reset" sans recalculer
    positions_init = np.asarray(positions_np, dtype=np.float32).copy()
    velocities_init = np.asarray(velocities_np, dtype=np.float32).copy()

    # Indices tissu
    indices_np = np.asarray(make_grid_line_indices(W, H, diagonals=True), dtype=np.uint32)
    tri_indices_np = np.asarray(make_grid_indices(W, H), dtype=np.uint32)

    positions_np = np.asarray(positions_np, dtype=np.float32)
    velocities_np = np.asarray(velocities_np, dtype=np.float32)

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
    # Depth buffer (taille dynamique = taille réelle de la swapchain)
    # ------------------------------------------------------------
    DEPTH_FORMAT = wgpu.TextureFormat.depth24plus
    depth_tex = None
    depth_view = None
    depth_size = (0, 0)

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

    # Normales (compute -> rendu)
    normal_buf = device.create_buffer(
        size=positions_np.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    # Index buffers tissu
    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )
    tri_idx_buf = device.create_buffer_with_data(
        data=tri_indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # ------------------------------------------------------------
    # 6) Sphère : wireframe + triangles (surface)
    # ------------------------------------------------------------
    # Wireframe
    sphere_pos_np, sphere_idx_np = make_uv_sphere_wire(stacks=16, slices=32)
    sphere_pos_buf = device.create_buffer_with_data(
        data=np.asarray(sphere_pos_np, dtype=np.float32).tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    sphere_idx_buf = device.create_buffer_with_data(
        data=np.asarray(sphere_idx_np, dtype=np.uint32).tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST,
    )

    # Surface triangles
    sphere_tri_pos_np, sphere_tri_idx_np = make_uv_sphere_triangles(stacks=16, slices=32)
    sphere_tri_pos_buf = device.create_buffer_with_data(
        data=np.asarray(sphere_tri_pos_np, dtype=np.float32).tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )
    sphere_tri_idx_buf = device.create_buffer_with_data(
        data=np.asarray(sphere_tri_idx_np, dtype=np.uint32).tobytes(),
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

    sphere_renderer = SphereRenderer(canvas, device, index_count=sphere_idx_np.size)          # wire
    sphere_renderer_lit = SphereRendererLit(canvas, device, index_count=sphere_tri_idx_np.size)  # surface

    print("✅ Renderers OK")

    # ------------------------------------------------------------
    # 9) Camera ORBIT (souris) — autour de la sphère
    # ------------------------------------------------------------
    aspect = 900 / 700
    model = np.eye(4, dtype=np.float32)

    target = np.array([sphere_cx, sphere_cy, sphere_cz], dtype=np.float32)

    # Etat caméra (orbit)
    cam_yaw   = 0.0        # rotation horizontale (radians)
    cam_pitch = 0.25       # rotation verticale (radians)
    cam_dist  = 4.5        # distance caméra -> target

    # limites + vitesses
    PITCH_MIN = -1.2
    PITCH_MAX =  1.2
    DIST_MIN  =  1.5
    DIST_MAX  = 10.0

    ROT_SPEED = 0.006      # sensibilité souris
    ZOOM_SPEED = 0.15      # sensibilité molette

    # drag state
    dragging = False
    last_x = None
    last_y = None

    def clamp(v, a, b):
        return a if v < a else b if v > b else v

    def compute_eye_from_orbit():
        """Convertit (yaw,pitch,dist) en position eye 3D autour de target."""
        cy = np.cos(cam_yaw);   sy = np.sin(cam_yaw)
        cp = np.cos(cam_pitch); sp = np.sin(cam_pitch)

        # direction depuis target vers la caméra
        # (yaw autour de Y, pitch autour de X local)
        dir_x = sy * cp
        dir_y = sp
        dir_z = cy * cp

        eye = target + cam_dist * np.array([dir_x, dir_y, dir_z], dtype=np.float32)
        return (float(eye[0]), float(eye[1]), float(eye[2]))

    def update_mvp():
        """Recalcule MVP et l'envoie aux renderers (tissu + sphère)."""
        eye = compute_eye_from_orbit()
        view = look_at(eye, (float(target[0]), float(target[1]), float(target[2])), (0.0, 1.0, 0.0))
        proj = perspective(70.0, aspect, 0.05, 50.0)
        mvp = proj @ view @ model
        mvp_bytes = mvp.T.astype(np.float32).tobytes()

        renderer_lit.set_mvp(mvp_bytes)
        renderer_wire.set_mvp(mvp_bytes)
        sphere_renderer.set_mvp(mvp_bytes)
        sphere_renderer_lit.set_mvp(mvp_bytes)

    # init MVP
    update_mvp()
    print("✅ Camera MVP set (orbit)")

    # ------------------------------------------------------------
    # Handlers souris (RenderCanvas pointer_*)
    # ------------------------------------------------------------
    def on_pointer_down(evt):
        nonlocal dragging, last_x, last_y
        dragging = True
        last_x = evt.get("x", None)
        last_y = evt.get("y", None)

    def on_pointer_up(evt):
        nonlocal dragging, last_x, last_y
        dragging = False
        last_x = None
        last_y = None

    def on_pointer_move(evt):
        nonlocal cam_yaw, cam_pitch, dragging, last_x, last_y
        if not dragging:
            return

        x = evt.get("x", None)
        y = evt.get("y", None)
        if x is None or y is None or last_x is None or last_y is None:
            last_x, last_y = x, y
            return

        dx = x - last_x
        dy = y - last_y
        last_x, last_y = x, y

        # yaw à gauche/droite, pitch en haut/bas
        cam_yaw   += dx * ROT_SPEED
        cam_pitch += (-dy) * ROT_SPEED
        cam_pitch = clamp(cam_pitch, PITCH_MIN, PITCH_MAX)

        update_mvp()

    def on_wheel(evt):
        nonlocal cam_dist
        # selon versions: wheel peut être "dy" ou "delta_y"
        dy = evt.get("dy", evt.get("delta_y", 0.0))
        if dy is None:
            dy = 0.0

        # convention: dy > 0 = scroll down => zoom out
        cam_dist *= (1.0 + float(dy) * ZOOM_SPEED * 0.01)
        cam_dist = clamp(cam_dist, DIST_MIN, DIST_MAX)

        update_mvp()

    # Hook events
    canvas.add_event_handler(on_pointer_down, "pointer_down")
    canvas.add_event_handler(on_pointer_up, "pointer_up")
    canvas.add_event_handler(on_pointer_move, "pointer_move")
    canvas.add_event_handler(on_wheel, "wheel")
    print("✅ Orbit controls: drag = rotate, wheel = zoom")


    # ------------------------------------------------------------
    # 10) Compute pipelines
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
    # 11) "Boutons" clavier / toggles (sans UI)
    # ------------------------------------------------------------
    paused = False
    show_cloth_surface = True
    show_cloth_wire = True
    show_sphere_surface = True
    show_sphere_wire = False

    print("\n🎛️  Contrôles clavier :")
    print("  P : pause / resume")
    print("  R : reset (tissu au départ)")
    print("  1 : toggle tissu surface")
    print("  2 : toggle tissu wireframe")
    print("  3 : toggle sphère surface")
    print("  4 : toggle sphère wireframe")
    print("  H : réaffiche l'aide\n")

    def reset_sim():
        """Remet pos/vel à l'état initial (dans pos_a/vel_a + clear pos_b/vel_b)."""
        nonlocal paused
        paused = False
        device.queue.write_buffer(pos_a, 0, positions_init.tobytes())
        device.queue.write_buffer(vel_a, 0, velocities_init.tobytes())
        # On copie aussi dans B pour éviter un 'jump' au prochain ping-pong
        device.queue.write_buffer(pos_b, 0, positions_init.tobytes())
        device.queue.write_buffer(vel_b, 0, velocities_init.tobytes())
        print("🔁 Reset OK")

    # RenderCanvas : selon versions, l'API event peut varier.
    # On essaye plusieurs options sans crasher.
    def on_key(evt):
        nonlocal paused, show_cloth_surface, show_cloth_wire, show_sphere_surface, show_sphere_wire
        key = (evt.get("key") or evt.get("text") or "").lower()

        if key == "p":
            paused = not paused
            print("⏸️ Pause" if paused else "▶️ Resume")
        elif key == "r":
            reset_sim()
        elif key == "1":
            show_cloth_surface = not show_cloth_surface
            print("Tissu surface:", show_cloth_surface)
        elif key == "2":
            show_cloth_wire = not show_cloth_wire
            print("Tissu wire:", show_cloth_wire)
        elif key == "3":
            show_sphere_surface = not show_sphere_surface
            print("Sphère surface:", show_sphere_surface)
        elif key == "4":
            show_sphere_wire = not show_sphere_wire
            print("Sphère wire:", show_sphere_wire)
        elif key == "h":
            print("\n🎛️  Contrôles clavier :")
            print("  P : pause / resume")
            print("  R : reset (tissu au départ)")
            print("  1 : toggle tissu surface")
            print("  2 : toggle tissu wireframe")
            print("  3 : toggle sphère surface")
            print("  4 : toggle sphère wireframe\n")

    # ------------------------------------------------------------
    # Hook clavier (RenderCanvas API varie selon versions)
    # ------------------------------------------------------------
    def on_any_event(evt):
        """
        evt est souvent un dict qui contient un type et des infos clavier.
        On essaye de détecter un "key down" et d'extraire la touche.
        """
        # evt peut être un dict ou un objet
        if isinstance(evt, dict):
            etype = (evt.get("type") or evt.get("event_type") or "").lower()
            key = (evt.get("key") or evt.get("text") or evt.get("value") or "").lower()
        else:
            # fallback objet
            etype = (getattr(evt, "type", "") or getattr(evt, "event_type", "") or "").lower()
            key = (getattr(evt, "key", "") or getattr(evt, "text", "") or getattr(evt, "value", "") or "").lower()

        # On filtre : on ne veut que les pressions clavier
        if "key" in etype and ("down" in etype or etype in ("key_down", "keydown", "key")):
            if key:
                on_key({"key": key})

    hooked = False

    # ✅ Cas 1 : add_event_handler(callback) -> reçoit tous les events
    try:
        canvas.add_event_handler(on_any_event)
        hooked = True
        print("✅ Keyboard events hooked: canvas.add_event_handler(callback)")
    except Exception:
        pass

    # ✅ Cas 2 : add_event_handler(event_type, callback)
    if not hooked:
        for name in ("key_down", "keydown", "key"):
            try:
                canvas.add_event_handler(name, on_key)
                hooked = True
                print(f"✅ Keyboard events hooked: canvas.add_event_handler('{name}', callback)")
                break
            except Exception:
                pass

    # ✅ Cas 3 : add_event_handler(callback, event_type)
    if not hooked:
        for name in ("key_down", "keydown", "key"):
            try:
                canvas.add_event_handler(on_key, name)
                hooked = True
                print(f"✅ Keyboard events hooked: canvas.add_event_handler(callback, '{name}')")
                break
            except Exception:
                pass

    if not hooked:
        print("❌ Clavier: impossible de hook les events (API RenderCanvas incompatible).")
        print("➡️ Dis-moi la version de rendercanvas (pip show rendercanvas) et je te donne le hook exact.")


    # ------------------------------------------------------------
    # 12) Helper : appeler encode() avec ou sans depth_view
    # ------------------------------------------------------------
    def call_encode(renderer, *args, depth_view=None, clear=False):
        """
        Certains de tes encode() acceptent (depth_view, clear),
        d'autres seulement (clear). On détecte et on appelle correctement.
        """
        try:
            sig = inspect.signature(renderer.encode)
            params = list(sig.parameters.keys())
            # On teste la présence de depth_view dans la signature
            if "depth_view" in params:
                return renderer.encode(*args, depth_view, clear=clear)
            else:
                return renderer.encode(*args, clear=clear)
        except Exception:
            # fallback simple : on tente sans depth
            return renderer.encode(*args, clear=clear)

    # ------------------------------------------------------------
    # 13) Boucle animation
    # ------------------------------------------------------------
    frame = 0
    ping = True
    dt_sub = np.float32(DT / SUBSTEPS)

    @canvas.request_draw
    def draw_frame():
        nonlocal frame, ping, depth_tex, depth_view, depth_size, paused
        frame += 1

        cx, cy, cz = sphere_cx, sphere_cy, sphere_cz
        r = sphere_r

        # =========================================================
        # 1) PHYSIQUE (SUBSTEPS) — seulement si pas en pause
        # =========================================================
        if not paused:
            for _ in range(SUBSTEPS):
                # --- Springs ---
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

                # --- Collision sphere + friction + sol ---
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

        # Buffer courant après ping-pong
        current_pos = pos_a if ping else pos_b

        # =========================================================
        # 2) NORMALES (toujours, même en pause : rendu correct)
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
        # 3) RENDU + depth dynamique
        # =========================================================
        tex = context.get_current_texture()
        view_tex = tex.create_view()
        enc = device.create_command_encoder()

        # IMPORTANT: depth doit matcher la taille réelle du swapchain (DPI)
        w = tex.width
        h = tex.height
        if depth_tex is None or depth_size != (w, h):
            depth_tex = device.create_texture(
                size=(w, h, 1),
                format=DEPTH_FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            depth_view = depth_tex.create_view()
            depth_size = (w, h)
            print(f"✅ depth resized to {w}x{h}")

        # Sphère rendue = sphère physique
        sphere_renderer.set_sphere((cx, cy, cz), r)
        sphere_renderer_lit.set_sphere((cx, cy, cz), r)

        # --- Ordre conseillé (si depth est actif) ---
        # 1) clear: tissu surface (ou sphère surface, c'est OK tant que clear=True une seule fois)
        cleared = False

        if show_cloth_surface:
            call_encode(renderer_lit, enc, view_tex, current_pos, normal_buf, tri_idx_buf, depth_view=depth_view, clear=True)
            cleared = True
        else:
            # Si on ne dessine pas le tissu surface, on clear au moins l'écran via la sphère surface
            if show_sphere_surface:
                call_encode(sphere_renderer_lit, enc, view_tex, sphere_tri_pos_buf, sphere_tri_idx_buf, depth_view=depth_view, clear=True)
                cleared = True
            else:
                # Sinon on clear "vide" en utilisant le wire tissu (load_op.clear) -> pas idéal,
                # mais on reste compatible sans toucher aux renderers.
                renderer_wire.encode(enc, view_tex, current_pos, idx_buf, clear=True)
                cleared = True

        # 2) sphère surface (par-dessus, depth gère l'occlusion)
        if show_sphere_surface:
            call_encode(sphere_renderer_lit, enc, view_tex, sphere_tri_pos_buf, sphere_tri_idx_buf, depth_view=depth_view, clear=False)

        # 3) overlays wireframe
        if show_cloth_wire:
            renderer_wire.encode(enc, view_tex, current_pos, idx_buf, clear=False)

        if show_sphere_wire:
            sphere_renderer.encode(enc, view_tex, sphere_pos_buf, sphere_idx_buf, clear=False)

        device.queue.submit([enc.finish()])

        if frame % 60 == 0:
            print("frame", frame)

        canvas.request_draw()

    loop.run()


if __name__ == "__main__":
    main()
