import wgpu
import numpy as np
from .gpu_utils import read_text


class SphereRendererLit:
    """
    Sphere surface renderer (triangles).
    - set_mvp(mvp_bytes)
    - set_sphere((cx,cy,cz), r)
    - encode(enc, color_view, sphere_pos_buf, sphere_tri_idx_buf, clear=False)
    """
    def __init__(self, canvas, device, index_count: int):
        self.canvas = canvas
        self.device = device
        self.queue = device.queue
        self.index_count = int(index_count)

        self.context = canvas.get_context("wgpu")
        self.texture_format = self.context.get_preferred_format(device.adapter)
        # IMPORTANT: NE PAS configure ici (fait dans main une seule fois)

        # ✅ shader lit (surface)
        shader_code = read_text("shaders/render_sphere_lit.wgsl")
        shader = device.create_shader_module(code=shader_code)

        # Camera = 64 bytes
        self.cam_buf = device.create_buffer(size=64, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.cam_bgl = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.cam_bg = device.create_bind_group(layout=self.cam_bgl, entries=[{
            "binding": 0,
            "resource": {"buffer": self.cam_buf, "offset": 0, "size": 64}
        }])

        # SphereU = 7 vec4 = 112 bytes (même format que le wire)
        self.sphere_buf = device.create_buffer(size=112, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.sphere_bgl = device.create_bind_group_layout(entries=[{
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        }])
        self.sphere_bg = device.create_bind_group(layout=self.sphere_bgl, entries=[{
            "binding": 0,
            "resource": {"buffer": self.sphere_buf, "offset": 0, "size": 112}
        }])

        pl = device.create_pipeline_layout(bind_group_layouts=[self.cam_bgl, self.sphere_bgl])

        # ✅ TRIANGLES (surface), pas line_list
        self.pipeline = device.create_render_pipeline(
            layout=pl,
            vertex={
                "module": shader,
                "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 16,  # vec4<f32>
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [{"shader_location": 0, "offset": 0, "format": wgpu.VertexFormat.float32x4}],
                }],
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [{"format": self.texture_format}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list, "cull_mode": wgpu.CullMode.back},
        )

        self.set_sphere((0.0, 0.8, 0.0), 0.6)

    def set_mvp(self, mvp_bytes: bytes):
        self.queue.write_buffer(self.cam_buf, 0, mvp_bytes)

    def set_sphere(self, center_xyz, radius: float):
        cx, cy, cz = center_xyz
        vecs = np.array([
            [cx, cy, cz, 0.0],
            [radius, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        self.queue.write_buffer(self.sphere_buf, 0, vecs.tobytes())

    def encode(self, enc, color_view, sphere_pos_buf, sphere_tri_idx_buf, clear: bool = False):
        rp = enc.begin_render_pass(color_attachments=[{
            "view": color_view,
            "load_op": wgpu.LoadOp.clear if clear else wgpu.LoadOp.load,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.1, 0.1, 0.1, 1.0),
        }])

        rp.set_pipeline(self.pipeline)
        rp.set_bind_group(0, self.cam_bg, [], 0, 999999)
        rp.set_bind_group(1, self.sphere_bg, [], 0, 999999)
        rp.set_vertex_buffer(0, sphere_pos_buf, 0)
        rp.set_index_buffer(sphere_tri_idx_buf, wgpu.IndexFormat.uint32, 0)
        rp.draw_indexed(self.index_count, 1, 0, 0, 0)
        rp.end()
