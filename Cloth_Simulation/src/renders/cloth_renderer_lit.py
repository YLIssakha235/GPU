import wgpu
from ..gpu_utils import read_text


class ClothRendererLit:
    def __init__(self, canvas, device, tri_index_count: int):
        self.canvas = canvas
        self.device = device
        self.queue = device.queue
        self.tri_index_count = int(tri_index_count)

        self.context = canvas.get_context("wgpu")
        self.texture_format = self.context.get_preferred_format(device.adapter)

        shader = device.create_shader_module(code=read_text("shaders/render_lit.wgsl"))

       
        # Camera uniform 
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
            entries=[{
                "binding": 0,
                "resource": {"buffer": self.cam_buf, "offset": 0, "size": 64},
            }],
        )

        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[self.cam_bgl])

        
        # Render pipeline (avec depth)
        self.pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [
                    # positions (location 0)
                    {
                        "array_stride": 16,
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [{
                            "shader_location": 0,
                            "offset": 0,
                            "format": wgpu.VertexFormat.float32x4,
                        }],
                    },
                    # normals (location 1)
                    {
                        "array_stride": 16,
                        "step_mode": wgpu.VertexStepMode.vertex,
                        "attributes": [{
                            "shader_location": 1,
                            "offset": 0,
                            "format": wgpu.VertexFormat.float32x4,
                        }],
                    },
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
            },

            # Depth test ON
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
        )

    def set_mvp(self, mvp: bytes):
        self.queue.write_buffer(self.cam_buf, 0, mvp)

    def encode(self, enc, color_view, position_buffer, normal_buffer, tri_index_buffer, depth_view, clear: bool = True):
        rp = enc.begin_render_pass(
            color_attachments=[{
                "view": color_view,
                "load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.1, 0.1, 0.1, 1.0),
            }],
            depth_stencil_attachment={
                "view": depth_view,
                "depth_load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )

        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.cam_bg, [], 0, 999999)
        rp.set_vertex_buffer(0, position_buffer, 0)
        rp.set_vertex_buffer(1, normal_buffer, 0)
        rp.set_index_buffer(tri_index_buffer, wgpu.IndexFormat.uint32, 0)
        rp.draw_indexed(self.tri_index_count, 1, 0, 0, 0)
        rp.end()
