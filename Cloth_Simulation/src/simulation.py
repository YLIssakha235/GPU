# src/simulation.py
"""
Simulation physique du tissu sur GPU (compute shaders).
- ressorts (structural / shear / bend)
- collisions sphère + sol (friction)
- ping-pong buffers
- calcul des normales
"""

import numpy as np
import wgpu

from src.data_init import make_grid_cloth


class ClothSimulation:
    def __init__(self, device):
        self.device = device

        # ===============================
        # PARAMÈTRES PHYSIQUES
        # ===============================
        self.G = -9.81

        self.K_STRUCT = 60.0
        self.K_SHEAR = 100.0
        self.K_BEND = 400.0

        self.DAMPING = 0.995

        self.DT = 1 / 240
        self.SUBSTEPS = 20
        self.REST = 0.10
        self.MASS = 0.5

        self.WORKGROUP_SIZE = 64

        # ===============================
        # SPHÈRE / SOL
        # ===============================
        self.SPHERE_R = 0.8
        self.MU = 0.8
        self.EPS = 0.01
        self.BOUNCE = 0.0
        self.FLOOR_Y = -2.0

        # ===============================
        # INIT MESH + BUFFERS + PIPELINES
        # ===============================
        self._init_mesh()
        self._init_buffers()
        self._init_pipelines()

    # ------------------------------------------------------------
    # INIT CPU : tissu
    # ------------------------------------------------------------
    def _init_mesh(self):
        # Grille tissu
        self.W, self.H = 16 , 16  # nb de points

        # Sphère (source de vérité de la scène)
        self.sphere_cx, self.sphere_cy, self.sphere_cz = 0.35, 1.0, 0.0

        # Position initiale du tissu AU-DESSUS de la sphère
        cloth_y0 = self.sphere_cy + self.SPHERE_R + 0.50

        pos, vel = make_grid_cloth(
            self.W, self.H, self.REST,
            y0=cloth_y0,
            cx=self.sphere_cx,
            cz=self.sphere_cz,
        )
        vel[:] = 0.0

        # Copies CPU pour reset
        self.positions_init = np.asarray(pos, dtype=np.float32).copy()
        self.velocities_init = np.asarray(vel, dtype=np.float32).copy()

        self.positions_np = self.positions_init.copy()
        self.velocities_np = self.velocities_init.copy()

        self.N = int(self.positions_np.shape[0])

        # Dispatch compute
        self.dispatch_x = (self.N + self.WORKGROUP_SIZE - 1) // self.WORKGROUP_SIZE

    # ------------------------------------------------------------
    # BUFFERS GPU
    # ------------------------------------------------------------
    def _init_buffers(self):
        d = self.device

        # Positions ping-pong (STORAGE + VERTEX)
        self.pos_a = d.create_buffer_with_data(
            data=self.positions_np.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self.pos_b = d.create_buffer(
            size=self.positions_np.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Vitesses ping-pong (STORAGE)
        self.vel_a = d.create_buffer_with_data(
            data=self.velocities_np.tobytes(),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.vel_b = d.create_buffer(
            size=self.velocities_np.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Normales (compute -> rendu) : STORAGE + VERTEX
        self.normal_buf = d.create_buffer(
            size=self.positions_np.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Ping = True => "A est courant"
        self.ping = True

    # ------------------------------------------------------------
    # PIPELINES COMPUTE
    # ------------------------------------------------------------
    def _init_pipelines(self):
        d = self.device

        # =========================================================
        # A) SPRINGS (struct + shear + bend)
        # =========================================================
        self.params_springs = d.create_buffer(
            size=48,  # 3 blocs (2x vec4<f32> + 1x vec4<u32>) = 48 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        springs_code = open("shaders/step2_structural_shear_bend.wgsl", encoding="utf-8").read()
        springs_mod = d.create_shader_module(code=springs_code)

        springs_bgl = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        ])

        # 0 = A->B, 1 = B->A
        self.bg_springs = [
            d.create_bind_group(layout=springs_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_a}},
                {"binding": 1, "resource": {"buffer": self.vel_a}},
                {"binding": 2, "resource": {"buffer": self.pos_b}},
                {"binding": 3, "resource": {"buffer": self.vel_b}},
                {"binding": 4, "resource": {"buffer": self.params_springs}},
            ]),
            d.create_bind_group(layout=springs_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_b}},
                {"binding": 1, "resource": {"buffer": self.vel_b}},
                {"binding": 2, "resource": {"buffer": self.pos_a}},
                {"binding": 3, "resource": {"buffer": self.vel_a}},
                {"binding": 4, "resource": {"buffer": self.params_springs}},
            ]),
        ]

        self.pipeline_springs = d.create_compute_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[springs_bgl]),
            compute={"module": springs_mod, "entry_point": "main"},
        )

        # =========================================================
        # B) COLLISION (sphère + friction + sol)
        # =========================================================
        self.params_collision = d.create_buffer(
            size=64,  # 12 floats (48 bytes) + 4 u32 (16 bytes)
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        collision_code = open("shaders/step4_collision_friction.wgsl", encoding="utf-8").read()
        collision_mod = d.create_shader_module(code=collision_code)

        collision_bgl = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        ])

        self.bg_collision = [
            d.create_bind_group(layout=collision_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_a}},
                {"binding": 1, "resource": {"buffer": self.vel_a}},
                {"binding": 2, "resource": {"buffer": self.pos_b}},
                {"binding": 3, "resource": {"buffer": self.vel_b}},
                {"binding": 4, "resource": {"buffer": self.params_collision}},
            ]),
            d.create_bind_group(layout=collision_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_b}},
                {"binding": 1, "resource": {"buffer": self.vel_b}},
                {"binding": 2, "resource": {"buffer": self.pos_a}},
                {"binding": 3, "resource": {"buffer": self.vel_a}},
                {"binding": 4, "resource": {"buffer": self.params_collision}},
            ]),
        ]

        self.pipeline_collision = d.create_compute_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[collision_bgl]),
            compute={"module": collision_mod, "entry_point": "main"},
        )

        # =========================================================
        # C) NORMALES (sur grille)
        # =========================================================
        self.params_normals = d.create_buffer(
            size=16,  # vec4<u32> : (W,H,0,0)
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        normals_code = open("shaders/compute_normals_grid.wgsl", encoding="utf-8").read()
        normals_mod = d.create_shader_module(code=normals_code)

        normals_bgl = d.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},
        ])

        self.bg_normals = [
            d.create_bind_group(layout=normals_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_a}},
                {"binding": 1, "resource": {"buffer": self.normal_buf}},
                {"binding": 2, "resource": {"buffer": self.params_normals}},
            ]),
            d.create_bind_group(layout=normals_bgl, entries=[
                {"binding": 0, "resource": {"buffer": self.pos_b}},
                {"binding": 1, "resource": {"buffer": self.normal_buf}},
                {"binding": 2, "resource": {"buffer": self.params_normals}},
            ]),
        ]

        self.pipeline_normals = d.create_compute_pipeline(
            layout=d.create_pipeline_layout(bind_group_layouts=[normals_bgl]),
            compute={"module": normals_mod, "entry_point": "main"},
        )

    # ------------------------------------------------------------
    # API PUBLIQUE
    # ------------------------------------------------------------
    def reset(self):
        """Réinitialise le tissu à l'état initial."""
        q = self.device.queue
        q.write_buffer(self.pos_a, 0, self.positions_init.tobytes())
        q.write_buffer(self.vel_a, 0, self.velocities_init.tobytes())
        q.write_buffer(self.pos_b, 0, self.positions_init.tobytes())
        q.write_buffer(self.vel_b, 0, self.velocities_init.tobytes())
        self.ping = True

    def step(self):
        """
        Avance la simulation d'une frame.
        IMPORTANT : on reproduit exactement la logique du brouillon :
        - springs (ping)
        - collision (ping)
        pour chaque substep
        """
        dt_sub = np.float32(self.DT / self.SUBSTEPS)

        for _ in range(self.SUBSTEPS):

            # =====================================================
            # 1) SPRINGS
            # =====================================================
            springs_params = b"".join([
                np.array([dt_sub, self.G, self.REST, self.MASS], dtype=np.float32).tobytes(),
                np.array([self.K_STRUCT, self.K_SHEAR, self.K_BEND, self.DAMPING], dtype=np.float32).tobytes(),
                np.array([self.W, self.H, self.N, 0], dtype=np.uint32).tobytes(),
            ])
            self.device.queue.write_buffer(self.params_springs, 0, springs_params)

            bg = self.bg_springs[0 if self.ping else 1]

            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipeline_springs)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(self.dispatch_x)
            cp.end()
            self.device.queue.submit([enc.finish()])

            self.ping = not self.ping

            # =====================================================
            # 2) COLLISION (sphère + friction + sol)
            # =====================================================
            collision_params = b"".join([
                np.array([
                    dt_sub,
                    self.sphere_cx, self.sphere_cy, self.sphere_cz,
                    self.SPHERE_R, self.BOUNCE, self.MU, self.EPS,
                    self.FLOOR_Y, 0.0, 0.0, 0.0
                ], dtype=np.float32).tobytes(),
                np.array([self.N, 0, 0, 0], dtype=np.uint32).tobytes(),
            ])
            self.device.queue.write_buffer(self.params_collision, 0, collision_params)

            bg = self.bg_collision[0 if self.ping else 1]

            enc = self.device.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(self.pipeline_collision)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups(self.dispatch_x)
            cp.end()
            self.device.queue.submit([enc.finish()])

            self.ping = not self.ping

    def compute_normals(self):
        """Recalcule les normales (à appeler chaque frame, même en pause)."""
        self.device.queue.write_buffer(
            self.params_normals,
            0,
            np.array([self.W, self.H, 0, 0], dtype=np.uint32).tobytes()
        )

        bg = self.bg_normals[0 if self.ping else 1]

        enc = self.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(self.pipeline_normals)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups(self.dispatch_x)
        cp.end()
        self.device.queue.submit([enc.finish()])

    @property
    def current_pos_buffer(self):
        """Buffer position courant après ping-pong."""
        return self.pos_a if self.ping else self.pos_b
