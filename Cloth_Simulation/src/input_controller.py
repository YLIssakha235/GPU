# src/input_controller.py
"""
Gestion des entrées utilisateur :
- souris : orbit caméra + zoom
- clavier : pause, reset, toggles d'affichage
"""

class InputController:
    def __init__(self, canvas, simulation, camera_scene):
        self.canvas = canvas
        self.simulation = simulation
        self.scene = camera_scene

        self.dragging = False
        self.last_x = None
        self.last_y = None

        self.paused = False

        self._hook_mouse()
        self._hook_keyboard()

        self._print_help()

    # ------------------------------------------------------------
    # SOURIS
    # ------------------------------------------------------------
    def _hook_mouse(self):
        # pass # souris désactivée
        self.canvas.add_event_handler(self.on_pointer_down, "pointer_down")
        self.canvas.add_event_handler(self.on_pointer_up, "pointer_up")
        self.canvas.add_event_handler(self.on_pointer_move, "pointer_move")
        self.canvas.add_event_handler(self.on_wheel, "wheel")

    def on_pointer_down(self, evt):
        self.dragging = True
        self.last_x = evt.get("x")
        self.last_y = evt.get("y")

    def on_pointer_up(self, evt):
        self.dragging = False
        self.last_x = None
        self.last_y = None

    def on_pointer_move(self, evt):
        if not self.dragging:
            return

        x, y = evt.get("x"), evt.get("y")
        if x is None or y is None:
            return

        dx = x - self.last_x
        dy = y - self.last_y

        self.last_x, self.last_y = x, y

        self.scene.cam_yaw += dx * self.scene.ROT_SPEED
        self.scene.cam_pitch += (-dy) * self.scene.ROT_SPEED
        self.scene.cam_pitch = self.scene.clamp(
            self.scene.cam_pitch,
            self.scene.PITCH_MIN,
            self.scene.PITCH_MAX,
        )

        self.scene.update_mvp()

    def on_wheel(self, evt):
        dy = evt.get("dy", evt.get("delta_y", 0.0))
        self.scene.cam_dist *= (1.0 + float(dy) * self.scene.ZOOM_SPEED * 0.01)
        self.scene.cam_dist = self.scene.clamp(
            self.scene.cam_dist,
            self.scene.DIST_MIN,
            self.scene.DIST_MAX,
        )
        self.scene.update_mvp()

    # ------------------------------------------------------------
    # CLAVIER
    # ------------------------------------------------------------
    def _hook_keyboard(self):
        # pass # clavier désactivé
        # Événements clavier (RenderCanvas)
        self.canvas.add_event_handler(self.on_any_event, "key_down")


    def on_any_event(self, evt):
        key = (evt.get("key") or evt.get("text") or "").lower()
        if not key:
            return

        if key == "p":
            self.paused = not self.paused
            print("⏸️ Pause" if self.paused else "▶️ Resume")

        elif key == "r":
            self.simulation.reset()
            self.paused = False
            print("🔁 Reset")

        elif key == "1":
            self.scene.show_cloth_surface = not self.scene.show_cloth_surface

        elif key == "2":
            self.scene.show_cloth_wire = not self.scene.show_cloth_wire

        elif key == "3":
            self.scene.show_sphere_surface = not self.scene.show_sphere_surface

        elif key == "4":
            self.scene.show_sphere_wire = not self.scene.show_sphere_wire

        elif key == "h":
            self._print_help()

    # ------------------------------------------------------------
    # AIDE
    # ------------------------------------------------------------
    def _print_help(self):
        print("\n🎛️  Contrôles :")
        print("  P : pause / reprise")
        print("  R : reset")
        print("  1 : tissu surface")
        print("  2 : tissu wireframe")
        print("  3 : sphère surface")
        print("  4 : sphère wireframe")
        print("  souris : orbit caméra")
        print("  molette : zoom\n")
