import wgpu
from .gpu_utils import read_text


class ClothRenderer:
    def __init__(self, canvas, device, index_count: int, wireframe: bool = True):
        self.canvas = canvas
        self.device = device
        self.queue = device.queue
        self.index_count = int(index_count)
        self.wireframe = bool(wireframe)

        self.context = canvas.get_context("wgpu")
        self.texture_format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=self.texture_format)

        shader = device.create_shader_module(code=read_text("shaders/render_basic.wgsl"))

        # --- Camera uniform (mat4x4 = 64 bytes) ---
        self.cam_buf = device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.cam_bgl = device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }
        ])

        self.cam_bg = device.create_bind_group(
            layout=self.cam_bgl,
            entries=[{"binding": 0, "resource": {"buffer": self.cam_buf, "offset": 0, "size": 64}}],
        )

        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self.cam_bgl])

        topology = wgpu.PrimitiveTopology.line_list if self.wireframe else wgpu.PrimitiveTopology.triangle_list

        self.render_pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 16,  # vec4<f32>
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [{"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x4}],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={"topology": topology, "cull_mode": wgpu.CullMode.none},
        )

    def set_mvp(self, mvp: bytes):
        # mvp doit faire 64 bytes
        self.queue.write_buffer(self.cam_buf, 0, mvp)

    def draw(self, position_buffer, index_buffer):
        tex = self.context.get_current_texture()
        enc = self.device.create_command_encoder()

        rp = enc.begin_render_pass(color_attachments=[{
            "view": tex.create_view(),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.1, 0.1, 0.1, 1.0),
        }])

        rp.set_pipeline(self.render_pipeline)
        rp.set_bind_group(0, self.cam_bg, [], 0, 999999)
        rp.set_vertex_buffer(0, position_buffer, 0)
        rp.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32, 0)
        rp.draw_indexed(self.index_count, 1, 0, 0, 0)
        rp.end()

        self.queue.submit([enc.finish()])
