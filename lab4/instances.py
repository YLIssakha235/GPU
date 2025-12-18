import os
import numpy as np
import wgpu as wgpu
from scipy.spatial.transform import Rotation
from rendercanvas.auto import RenderCanvas, loop

class App:

  # ...

  def __init__(self):
    # Initialize GPU device and canvas early so attributes exist for
    # subsequent buffer/pipeline creation.
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    self.device = adapter.request_device_sync()
    # create a simple canvas so the app can run the draw loop if needed
    try:
        self.canvas = RenderCanvas(size=(640, 480), title="Instances", update_mode="continuous")
        self.context = self.canvas.get_wgpu_context()
    except Exception:
        # canvas creation may fail in headless/test environments; keep going
        self.canvas = None
        self.context = None
    # ...
    self.instance_count = 10
    # ...
    instance_data = b""
    for _ in range(self.instance_count):
        trans = np.eye(4)
        trans[:3, 3] = np.random.uniform(-5, 5, 3)

        rot = np.eye(4)
        rot[:3, :3] = Rotation.random().as_matrix()

        instance_data += (trans @ rot).T.astype(np.float32).tobytes()

    self.instance_buffer = self.device.create_buffer_with_data(
        data=instance_data, usage=wgpu.BufferUsage.VERTEX
    )
    # ...
    instance_buffer_descriptor = {
        "array_stride": 4 * 4 * 4,
        "step_mode": wgpu.VertexStepMode.instance,
        "attributes": [
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 0,
                "shader_location": 3,
            },
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 4 * 4,
                "shader_location": 4,
            },
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 8 * 4,
                "shader_location": 5,
            },
            {
                "format": wgpu.VertexFormat.float32x4,
                "offset": 12 * 4,
                "shader_location": 6,
            },
        ],
    }

    # Ensure we have a vertex buffer descriptor matching the vertex layout
    # (position: vec3, normal: vec3, uv: vec2) used by the shader.
    vertex_buffer_descriptor = {
        "array_stride": 8 * 4,
        "step_mode": wgpu.VertexStepMode.vertex,
        "attributes": [
            {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 0},
            {"format": wgpu.VertexFormat.float32x3, "offset": 3 * 4, "shader_location": 1},
            {"format": wgpu.VertexFormat.float32x2, "offset": 6 * 4, "shader_location": 2},
        ],
    }

    # Load WGSL shader and create shader module
    shader_path = os.path.join(os.path.dirname(__file__), "shader.wgsl")
    if not os.path.exists(shader_path):
        # try parent folder (project layout may vary)
        shader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lab4', 'shader.wgsl'))
    if not os.path.exists(shader_path):
        shader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shader.wgsl'))

    with open(shader_path, "r", encoding="utf-8") as f:
        shader_source = f.read()

    shader_module = self.device.create_shader_module(code=shader_source)

    # Determine render texture format from the context if available
    if self.context is not None:
        render_texture_format = self.context.get_preferred_format(self.device.adapter)
    else:
        render_texture_format = wgpu.TextureFormat.rgba8unorm_srgb

    # Create a bind group layout that matches the shader's group(0) bindings
    bg_layout = self.device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
            {"binding": 1, "visibility": wgpu.ShaderStage.FRAGMENT, "texture": {}},
            {"binding": 2, "visibility": wgpu.ShaderStage.FRAGMENT, "sampler": {}},
        ]
    )

    p_layout = self.device.create_pipeline_layout(bind_group_layouts=[bg_layout])

    self.pipeline = self.device.create_render_pipeline(
        layout=p_layout,
        vertex={
            "module": shader_module,
            "entry_point": "vs_main",
            "buffers": [vertex_buffer_descriptor, instance_buffer_descriptor],
        },
        fragment={
            "module": shader_module,
            "entry_point": "fs_main",
            "targets": [{"format": render_texture_format}],
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
        multisample=None,
    )

  def loop(self):
    # ...
    render_pass.set_vertex_buffer(1, self.instance_buffer)
    # ...
    render_pass.draw_indexed(36, self.instance_count)


if __name__ == "__main__":
    # Allow running this file directly for quick testing.
    try:
        App().run()
    except Exception:
        # Fallback: if App has no run() method or run raises, try the
        # usual rendercanvas loop pattern (safe fallback for debugging).
        from rendercanvas.auto import loop
        app = App()
        if hasattr(app, "canvas") and hasattr(app, "loop"):
            app.canvas.request_draw(app.loop)
            loop.run()
        else:
            raise