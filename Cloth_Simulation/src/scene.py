# src/scene.py
"""
Gestion de la scène visible :
- rendu du tissu (surface + wireframe)
- rendu de la sphère (surface + wireframe)
- gestion de la caméra et du MVP
"""

import numpy as np
import inspect
import wgpu

from src.camera import look_at, perspective
from src.data_init import (
    make_grid_line_indices,
    make_grid_indices,
    make_uv_sphere_wire,
    make_uv_sphere_triangles,
)

from src.renders.cloth_renderer import ClothRenderer
from src.renders.cloth_renderer_lit import ClothRendererLit
from src.renders.sphere_renderer import SphereRenderer
from src.renders.sphere_renderer_lit import SphereRendererLit


class Scene:
    def __init__(self, canvas, device):
        self.device = device
        self.canvas = canvas

        # ===============================
        # FLAGS DE RENDU (toggles clavier)
        # ===============================
        self.show_cloth_surface = True
        self.show_cloth_wire = False
        self.show_sphere_surface = True
        self.show_sphere_wire = True

        # ===============================
        # CAMERA ORBIT
        # ===============================
        self._init_camera()
        self.camera = self # pour compatibilité avec InputController

        # ===============================
        # GEOMETRIE & RENDERERS
        # ===============================
        self._init_cloth_geometry()
        self._init_sphere_geometry()
        self._init_renderers(canvas, device)

    # ------------------------------------------------------------
    # CAMERA
    # ------------------------------------------------------------
    def _init_camera(self):
        self.aspect = 900 / 700
        self.model = np.eye(4, dtype=np.float32)

        # Cible (centre de la sphère)
        self.target = np.array([0.35, 1.0, 0.0], dtype=np.float32)

        self.cam_yaw = 0.0
        self.cam_pitch = 0.25
        self.cam_dist = 4.5

        self.PITCH_MIN = -1.2
        self.PITCH_MAX = 1.2
        self.DIST_MIN = 1.5
        self.DIST_MAX = 10.0

        self.ROT_SPEED = 0.006
        self.ZOOM_SPEED = 0.15

    def clamp(self, v, a, b):
        return a if v < a else b if v > b else v

    def compute_eye(self):
        cy = np.cos(self.cam_yaw)
        sy = np.sin(self.cam_yaw)
        cp = np.cos(self.cam_pitch)
        sp = np.sin(self.cam_pitch)

        dir_x = sy * cp
        dir_y = sp
        dir_z = cy * cp

        eye = self.target + self.cam_dist * np.array([dir_x, dir_y, dir_z], dtype=np.float32)
        return tuple(eye.tolist())

    def update_mvp(self):
        eye = self.compute_eye()
        view = look_at(eye, tuple(self.target), (0.0, 1.0, 0.0))
        proj = perspective(70.0, self.aspect, 0.05, 50.0)
        mvp = proj @ view @ self.model
        mvp_bytes = mvp.T.astype(np.float32).tobytes()

        # Envoi aux renderers
        self.renderer_lit.set_mvp(mvp_bytes)
        self.renderer_wire.set_mvp(mvp_bytes)
        self.sphere_renderer.set_mvp(mvp_bytes)
        self.sphere_renderer_lit.set_mvp(mvp_bytes)

    # ------------------------------------------------------------
    # GEOMETRIE
    # ------------------------------------------------------------
    def _init_cloth_geometry(self):
        W, H = 16, 16  # doit correspondre à simulation.py
        self.idx_np = np.asarray(make_grid_line_indices(W, H, diagonals=True), np.uint32)
        self.tri_idx_np = np.asarray(make_grid_indices(W, H), np.uint32)

        self.idx_buf = self.device.create_buffer_with_data(
            data=self.idx_np.tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )
        self.tri_idx_buf = self.device.create_buffer_with_data(
            data=self.tri_idx_np.tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )

    def _init_sphere_geometry(self):
        pos, idx = make_uv_sphere_wire(16, 32)
        self.sphere_pos_buf = self.device.create_buffer_with_data(
            data=np.asarray(pos, np.float32).tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )
        self.sphere_idx_buf = self.device.create_buffer_with_data(
            data=np.asarray(idx, np.uint32).tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )

        pos, idx = make_uv_sphere_triangles(16, 32)
        self.sphere_tri_pos_buf = self.device.create_buffer_with_data(
            data=np.asarray(pos, np.float32).tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )
        self.sphere_tri_idx_buf = self.device.create_buffer_with_data(
            data=np.asarray(idx, np.uint32).tobytes(),
            usage=wgpu.BufferUsage.INDEX,
        )

    # ------------------------------------------------------------
    # RENDERERS
    # ------------------------------------------------------------
    def _init_renderers(self, canvas, device):
        self.renderer_lit = ClothRendererLit(canvas, device, self.tri_idx_np.size)
        self.renderer_wire = ClothRenderer(canvas, device, self.idx_np.size)

        self.sphere_renderer = SphereRenderer(canvas, device, self.sphere_idx_buf.size // 4)
        self.sphere_renderer_lit = SphereRendererLit(canvas, device, self.sphere_tri_idx_buf.size // 4)

        self.update_mvp()

    # ------------------------------------------------------------
    # DRAW
    # ------------------------------------------------------------
    def _call_encode(self, renderer, *args, depth_view=None, clear=False):
        sig = inspect.signature(renderer.encode)
        if "depth_view" in sig.parameters:
            return renderer.encode(*args, depth_view, clear=clear)
        return renderer.encode(*args, clear=clear)

    def draw(self, device, view_tex, depth_view, sim):
        enc = device.create_command_encoder()

        # ✅ AJOUTE CES 2 LIGNES ICI
        self.sphere_renderer.set_sphere((sim.sphere_cx, sim.sphere_cy, sim.sphere_cz), sim.SPHERE_R)
        self.sphere_renderer_lit.set_sphere((sim.sphere_cx, sim.sphere_cy, sim.sphere_cz), sim.SPHERE_R)

        cleared = False

        if self.show_cloth_surface:
            self._call_encode(
                self.renderer_lit,
                enc, view_tex, sim.current_pos_buffer, sim.normal_buf, self.tri_idx_buf,
                depth_view=depth_view, clear=True
            )
            cleared = True

        if self.show_sphere_surface:
            self._call_encode(
                self.sphere_renderer_lit,
                enc, view_tex, self.sphere_tri_pos_buf, self.sphere_tri_idx_buf,
                depth_view=depth_view, clear=not cleared
            )

        if self.show_cloth_wire:
            self._call_encode(
                self.renderer_wire,
                enc, view_tex, sim.current_pos_buffer, self.idx_buf,
                depth_view=depth_view, clear=False
            )

        if self.show_sphere_wire:
            # ✅ CORRECTION: Ajout de depth_view
            self._call_encode(
                self.sphere_renderer,
                enc, view_tex, self.sphere_pos_buf, self.sphere_idx_buf,
                depth_view=depth_view, clear=False
            )

        device.queue.submit([enc.finish()])