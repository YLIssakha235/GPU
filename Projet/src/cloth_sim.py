import numpy as np
import wgpu
import struct
from .gpu_utils import create_uniform_buffer

class ClothSimulation:
    """Gère la simulation physique du tissu sur GPU (compute shaders)."""
    
    def __init__(self, device, positions_np, velocities_np, indices_np, W, H, params):
        """
        Args:
            device: GPU device
            positions_np: Array (N, 4) positions initiales
            velocities_np: Array (N, 4) vitesses initiales
            indices_np: Array indices pour le rendu
            W, H: Dimensions de la grille
            params: Dict avec {dt, g, k, rest, mass, damping, radius, mu, sphere_c}
        """
        self.device = device
        self.queue = device.queue
        self.W = W
        self.H = H
        self.N = W * H
        
        # Stockage des arrays numpy (pour tailles)
        self.positions_np = positions_np
        self.velocities_np = velocities_np
        self.indices_np = indices_np
        
        # Création des buffers GPU (double buffering)
        self._create_buffers()
        
        # Paramètres physiques
        self._create_params_buffer(params)
        
        # Compute pipelines pour chaque step
        self._create_compute_pipelines()
        
        # Workgroup size
        self.wg_size = 64
        self.num_workgroups = (self.N + self.wg_size - 1) // self.wg_size
        
    def _create_buffers(self):
        """Crée les buffers GPU avec double buffering."""
        # --- Force dtypes + bytes (IMPORTANT) ---
        pos_bytes = np.asarray(self.positions_np, dtype=np.float32).tobytes()
        vel_bytes = np.asarray(self.velocities_np, dtype=np.float32).tobytes()
        idx_bytes = np.asarray(self.indices_np, dtype=np.uint32).tobytes()

        # Usages
        pos_usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.VERTEX
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        vel_usage = (
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        )
        idx_usage = wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST

        # Buffers positions (double buffering)
        self.pos_in = self.device.create_buffer_with_data(
            data=pos_bytes,
            usage=pos_usage,
        )
        self.pos_out = self.device.create_buffer(
            size=len(pos_bytes),
            usage=pos_usage,
        )

        # Buffers vitesses (double buffering)
        self.vel_in = self.device.create_buffer_with_data(
            data=vel_bytes,
            usage=vel_usage,
        )
        self.vel_out = self.device.create_buffer(
            size=len(vel_bytes),
            usage=vel_usage,
        )

        # Buffer indices
        self.idx_buf = self.device.create_buffer_with_data(
            data=idx_bytes,
            usage=idx_usage,
        )

        
    def _create_params_buffer(self, params):
        """Crée le buffer uniform des paramètres physiques."""
        params_bytes = struct.pack(
            "ffffffffIIIIffff",
            float(params['dt']), float(params['g']), float(params['k']), float(params['rest']),
            float(params['mass']), float(params['damping']), float(params['radius']), float(params['mu']),
            int(self.W), int(self.H), int(self.N), 0,
            float(params['sphere_c'][0]), float(params['sphere_c'][1]), float(params['sphere_c'][2]), 0.0
        )
        self.params_buf = create_uniform_buffer(self.device, params_bytes)
        self.params_bytes = params_bytes
        
    def _create_compute_pipelines(self):
        """Crée tous les compute pipelines (1 par step physique)."""
        from .gpu_utils import read_text
        
        # Layout commun (tous les shaders ont la même structure de bind group)
        self.bind_group_layout = self.device.create_bind_group_layout(entries=[
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},  # pos_in
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "read-only-storage"}},  # vel_in
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},            # pos_out
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "storage"}},            # vel_out
            {"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": "uniform"}},            # params
        ])
        
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[self.bind_group_layout])
        
        # Liste des shaders à charger (dans l'ordre d'exécution)
        shader_files = [
            "shaders/step1_gravity.wgsl",
            "shaders/step2_structural.wgsl",
            "shaders/step3_collision_sphere.wgsl",
            "shaders/step4_collision_friction.wgsl",
            "shaders/step0_integrate.wgsl",
        ]
        
        self.pipelines = []
        for shader_path in shader_files:
            shader_code = read_text(shader_path)
            shader_module = self.device.create_shader_module(code=shader_code)
            pipeline = self.device.create_compute_pipeline(
                layout=pipeline_layout,
                compute={"module": shader_module, "entry_point": "main"},
            )
            self.pipelines.append(pipeline)
            
    def _create_bind_group(self, pos_in, vel_in, pos_out, vel_out):
        """Crée un bind group avec les buffers actuels."""
        return self.device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": pos_in, "offset": 0, "size": self.positions_np.nbytes}},
                {"binding": 1, "resource": {"buffer": vel_in, "offset": 0, "size": self.velocities_np.nbytes}},
                {"binding": 2, "resource": {"buffer": pos_out, "offset": 0, "size": self.positions_np.nbytes}},
                {"binding": 3, "resource": {"buffer": vel_out, "offset": 0, "size": self.velocities_np.nbytes}},
                {"binding": 4, "resource": {"buffer": self.params_buf, "offset": 0, "size": len(self.params_bytes)}},
            ],
        )
    
    def step(self):
        """Exécute un pas de simulation complet (tous les compute shaders)."""
        # Bind group avec buffers actuels
        bind_group = self._create_bind_group(self.pos_in, self.vel_in, self.pos_out, self.vel_out)
        
        encoder = self.device.create_command_encoder()
        
        # Exécuter tous les pipelines en séquence
        for pipeline in self.pipelines:
            compute_pass = encoder.begin_compute_pass()
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group, [])
            compute_pass.dispatch_workgroups(self.num_workgroups, 1, 1)
            compute_pass.end()
        
        self.queue.submit([encoder.finish()])
        
        # SWAP des buffers pour la frame suivante
        self.pos_in, self.pos_out = self.pos_out, self.pos_in
        self.vel_in, self.vel_out = self.vel_out, self.vel_in
    
    def get_position_buffer(self):
        """Retourne le buffer de positions actuel (pour le rendu)."""
        return self.pos_in
    
    def get_index_buffer(self):
        """Retourne le buffer d'indices."""
        return self.idx_buf
    
print("cloth_sim.py loaded.")