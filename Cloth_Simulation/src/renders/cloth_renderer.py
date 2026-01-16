import wgpu
from ..gpu_utils import read_text


class ClothRenderer:
    """
    Renderer wireframe du tissu (lignes).
    Utilisé pour le debug et l’overlay.
    Compatible depth (lecture seule).
    """

    def __init__(self, canvas, device, index_count: int):
        self.device = device 
        self.queue = device.queue 
        self.index_count = int(index_count) 

        context = canvas.get_context("wgpu")
        self.texture_format = context.get_preferred_format(device.adapter)

        shader = device.create_shader_module(
            code=read_text("shaders/render_basic.wgsl")
        )

        
        # Uniform caméra (MVP)
        self.cam_buf = device.create_buffer(
            size=64,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.cam_bgl = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])

        self.cam_bg = device.create_bind_group(
            layout=self.cam_bgl,
            entries=[{
                "binding": 0,
                "resource": {
                    "buffer": self.cam_buf,
                    "offset": 0,
                    "size": 64,
                },
            }],
        )

        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self.cam_bgl]
        )

        # Render pipeline wireframe depth lecture seule
        self.pipeline = device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 16,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [{
                        "shader_location": 0,
                        "offset": 0,
                        "format": wgpu.VertexFormat.float32x4,
                    }],
                }],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.line_list, # wireframe
                "cull_mode": wgpu.CullMode.none, # pas de culling
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": False,  # lecture seule
                "depth_compare": wgpu.CompareFunction.less,
            },
        )

    # API
    def set_mvp(self, mvp_bytes: bytes):
        self.queue.write_buffer(self.cam_buf, 0, mvp_bytes)

    def encode(
        self,
        enc,
        color_view,
        position_buffer,
        index_buffer,
        depth_view=None,
        clear: bool = False,
    ):
        attachments = {
            "color_attachments": [{
                "view": color_view,
                "load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0.1, 0.1, 0.1, 1.0),
            }]
        }

        if depth_view is not None:
            attachments["depth_stencil_attachment"] = {
                "view": depth_view,
                "depth_load_op": wgpu.LoadOp.load,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0, 
            }

        rp = enc.begin_render_pass(**attachments)
        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.cam_bg, [], 0, 999999)
        rp.set_vertex_buffer(0, position_buffer, 0)
        rp.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32, 0)
        rp.draw_indexed(self.index_count, 1, 0, 0, 0)
        rp.end()
