# src/app.py
"""
Point central de l'application.
- initialise GPU
- crée simulation + scène
- gère la boucle draw
"""

from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.simulation import ClothSimulation
from src.scene import Scene
from src.input_controller import InputController


def run_app():
    device = get_default_device()
    canvas = RenderCanvas(title="Cloth Simulation (refactor)", size=(900, 700))

    context = canvas.get_context("wgpu")
    format = context.get_preferred_format(device.adapter)
    context.configure(device=device, format=format)

    sim = ClothSimulation(device)
    scene = Scene(canvas, device)
    inputs = InputController(canvas, sim, scene.camera)

    depth_tex = None
    depth_view = None
    depth_size = (0, 0)

    @canvas.request_draw
    def draw():
        nonlocal depth_tex, depth_view, depth_size

        sim.step()
        sim.compute_normals()

        tex = context.get_current_texture()
        view = tex.create_view()

        if depth_tex is None or depth_size != (tex.width, tex.height):
            depth_tex = device.create_texture(
                size=(tex.width, tex.height, 1),
                format=wgpu.TextureFormat.depth24plus,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            depth_view = depth_tex.create_view()
            depth_size = (tex.width, tex.height)

        scene.draw(device, view, depth_view, sim)

        canvas.request_draw()

    loop.run()
