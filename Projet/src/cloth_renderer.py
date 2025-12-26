import wgpu
from .gpu_utils import read_text


class ClothRendererLit:
    def __init__(self, canvas, device, tri_index_count: int):
        self.canvas = canvas
        self.device = device
        self.queue = device.queue
        self.tri_index_count = int(tri_index_count)

        self.context = canvas.get_context("wgpu")
        self.texture_format = self.context.get_preferred_format(device.adapter)
        self.context.configure(device=device, format=self.texture_format)

        shader = device.create_shader_module(code=read_text("shaders/render_lit.wgsl"))

        # camera uniform
        self.cam_buf = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self.cam_bgl = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.cam_bg = device.create_bind_group(
            layout=self.cam_bgl,
            entries=[{"binding": 0, "resource": {"buffer": self.cam_buf, "offset": 0, "size": 64}}],
        )

        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self.cam_bgl])

        self.pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    # positions
                    {
                        "array_stride": 16,
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [{"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x4}],
                    },
                    # normals
                    {
                        "array_stride": 16,
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [{"shader_location": 1, "offset": 0, "format": wgpu.VertexFormat.float32x4}],
                    },
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.none},
        )

    def set_mvp(self, mvp_bytes: bytes):
        self.queue.write_buffer(self.cam_buf, 0, mvp_bytes)

    def draw(self, position_buffer, normal_buffer, tri_index_buffer):
        tex = self.context.get_current_texture()
        enc = self.device.create_command_encoder()

        rp = enc.begin_render_pass(color_attachments=[{
            "view": tex.create_view(),
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.1, 0.1, 0.1, 1.0),
        }])

        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.cam_bg, [], 0, 999999)
        rp.set_vertex_buffer(0, position_buffer, 0)
        rp.set_vertex_buffer(1, normal_buffer, 0)
        rp.set_index_buffer(tri_index_buffer, wgpu.IndexFormat.uint32, 0)
        rp.draw_indexed(self.tri_index_count, 1, 0, 0, 0)
        rp.end()

        self.queue.submit([enc.finish()])
