import numpy as np
from rendercanvas.auto import RenderCanvas, loop
from wgpu.utils import get_default_device
import wgpu

from src.data_init import make_grid_cloth, make_grid_line_indices
from src.renderer import ClothRenderer


def main():
    device = get_default_device()
    print("✅ Device:", device)

    # 1) Mesh CPU : grille (tissu)
    W, H = 20, 20
    positions_np, _vel = make_grid_cloth(W, H, 0.10)

    # indices wireframe (LINES)
    indices_np = make_grid_line_indices(W, H, diagonals=True)   # triangulation visible
    # indices_np = make_grid_line_indices(W, H, diagonals=False) # juste carrés


    # dtypes corrects
    positions_np = np.asarray(positions_np, dtype=np.float32)  # (N,4)
    indices_np = np.asarray(indices_np, dtype=np.uint32)       # (M,)

    print("pos x range:", positions_np[:, 0].min(), positions_np[:, 0].max())
    print("pos z range:", positions_np[:, 2].min(), positions_np[:, 2].max())
    print("✅ Mesh:", positions_np.shape, "line indices:", indices_np.shape)

    # 2) Canvas
    canvas = RenderCanvas(title="Cloth - Step2 (wireframe)", size=(900, 700))

    # 3) Buffers GPU
    pos_buf = device.create_buffer_with_data(
        data=positions_np.tobytes(),
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
    )
    idx_buf = device.create_buffer_with_data(
        data=indices_np.tobytes(),
        usage=wgpu.BufferUsage.INDEX | wgpu.BufferUsage.COPY_DST
    )

    # 4) Renderer (on passe juste le nombre d'indices)
    renderer = ClothRenderer(canvas, device, index_count=indices_np.size, wireframe=True)
    print("✅ Renderer OK")

    frame = 0

    @canvas.request_draw
    def draw_frame():
        nonlocal frame
        frame += 1
        if frame % 60 == 0:
            print("frame", frame)

        renderer.draw(pos_buf, idx_buf)
        canvas.request_draw()

    canvas.request_draw()
    loop.run()


if __name__ == "__main__":
    main()
