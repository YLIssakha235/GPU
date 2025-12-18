# ...
import random
from imgui_bundle import imgui  # Before rendercanvas
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils.imgui import ImguiRenderer

class App:

  # ...

  def __init__(self):
    # ...
    self.imgui_renderer = ImguiRenderer(self.device, self.canvas)

  def loop(self):
    # ...
    self.imgui_renderer.render()

  def update_gui(self):
    imgui.begin("Window", None)
    imgui.text("Example Text")
    if imgui.button("Hello"):
        print("World")
    imgui.end()

  def run(self):
    self.canvas.request_draw(self.loop)
    self.imgui_renderer.set_gui(self.update_gui)
    loop.run()