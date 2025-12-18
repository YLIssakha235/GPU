from rendercanvas.auto import RenderCanvas, loop
import wgpu
import numpy as np
import PIL.Image as Image
from camera import Camera
from primitives import sphere


class App:
    def __init__(self):
        # ================================
        # 1. Adapter / Device / Canvas
        # ================================
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        self.device = adapter.request_device_sync()

        self.size = (0, 0)

        self.canvas = RenderCanvas(
            size=(640, 480),
            title="Lab3 - Sphere",
            update_mode="continuous",
            max_fps=60,
        )
        self.context = self.canvas.get_wgpu_context()

        render_texture_format = self.context.get_preferred_format(adapter)
        self.context.configure(device=self.device, format=render_texture_format)

        # ================================
        # 2. Sphère : vertex + indices
        # ================================
        vertex_data, index_data = sphere()
        self.index_count = index_data.size  # <— IMPORTANT

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data, usage=wgpu.BufferUsage.VERTEX
        )
        self.index_buffer = self.device.create_buffer_with_data(
            data=index_data, usage=wgpu.BufferUsage.INDEX
        )

        # ================================
        # 3. Bind group layout (uniform + texture + sampler)
        # ================================
        bg_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX
                    | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {},
                },
            ]
        )

        p_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bg_layout]
        )

        # ================================
        # 4. Buffer uniforme (lumière + view + proj)
        # ================================
        self.render_params_buffer = self.device.create_buffer(
            size=4 * 4 + 2 * 4 * 4 * 4,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # ================================
        # 5. Texture simple 1x1 blanche
        #    (pas besoin de fichier PNG)
        # ================================
        img = Image.new("RGBA", (1, 1), (255, 255, 255, 255))
        texture_size = img.size + (1,)

        texture = self.device.create_texture(
            size=texture_size,
            format=wgpu.TextureFormat.rgba8unorm_srgb,
            usage=wgpu.TextureUsage.TEXTURE_BINDING
            | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            destination={"texture": texture},
            data=np.asarray(img),
            data_layout={
                "bytes_per_row": img.size[0] * 4,
                "rows_per_image": img.size[1],
            },
            size=texture_size,
        )

        sampler = self.device.create_sampler()

        self.render_params_bind_group = self.device.create_bind_group(
            layout=bg_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.render_params_buffer,
                        "offset": 0,
                        "size": self.render_params_buffer.size,
                    },
                },
                {
                    "binding": 1,
                    "resource": texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": sampler,
                },
            ],
        )

        # ================================
        # 6. Caméra orbitale
        # ================================
        self.camera = Camera(
            45, 640 / 480, 0.1, 100, 3, np.pi / 4, np.pi / 4
        )

        self.canvas.add_event_handler(
            self.process_event,
            "pointer_up",
            "pointer_down",
            "pointer_move",
            "wheel",
        )

        # ================================
        # 7. Description du vertex buffer
        # ================================
        vertex_buffer_descriptor = {
            "array_stride": 8 * 4,
            "step_mode": wgpu.VertexStepMode.vertex,
            "attributes": [
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 0,
                    "shader_location": 0,
                },
                {
                    "format": wgpu.VertexFormat.float32x3,
                    "offset": 3 * 4,
                    "shader_location": 1,
                },
                {
                    "format": wgpu.VertexFormat.float32x2,
                    "offset": 6 * 4,
                    "shader_location": 2,
                },
            ],
        }

        # ================================
        # 8. Charger le shader WGSL
        # ================================
        with open("render.wgsl") as file:
            shader_module = self.device.create_shader_module(
                code=file.read()
            )

        # ================================
        # 9. Pipeline de rendu
        # ================================
        self.pipeline = self.device.create_render_pipeline(
            layout=p_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [vertex_buffer_descriptor],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.back,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": render_texture_format,
                    }
                ],
            },
        )

        # sera créé dans loop au premier frame
        self.depth_texture = None

    # =====================================
    # Gestion des événements souris
    # =====================================
    def process_event(self, event):
        self.camera.process_event(event)

    # =====================================
    # Boucle de rendu
    # =====================================
    def loop(self):
        screen_texture: wgpu.GPUTexture = self.context.get_current_texture()  # type: ignore
        size = screen_texture.size

        if size[:2] != self.size:
            self.depth_texture = self.device.create_texture(
                size=size,
                format=wgpu.TextureFormat.depth32float,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self.size = size[:2]
            self.camera.aspect = size[0] / size[1]

        # lumière + matrices
        light_position = np.array([-10, 10, 10, 0], dtype=np.float32)
        proj_matrix, view_matrix = self.camera.get_matrices()

        render_params_data = light_position.tobytes()
        render_params_data += view_matrix.T.astype(np.float32).tobytes()
        render_params_data += proj_matrix.T.astype(np.float32).tobytes()

        self.device.queue.write_buffer(
            buffer=self.render_params_buffer,
            data=render_params_data,
            buffer_offset=0,
        )

        command_encoder = self.device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": screen_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0.1, 0.1, 0.1, 1.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_texture.create_view(),
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
            },
        )

        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_index_buffer(
            self.index_buffer, wgpu.IndexFormat.uint32
        )
        render_pass.set_bind_group(0, self.render_params_bind_group)
        render_pass.draw_indexed(self.index_count)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])

    def run(self):
        self.canvas.request_draw(self.loop)
        loop.run()


if __name__ == "__main__":
    App().run()
