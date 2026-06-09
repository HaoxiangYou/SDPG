from utils.terrain import Terrain
import numpy as np 
from PIL import Image
import os


terrain_cfg = {
    "mesh_type": "heightfield",
    "curriculum": True,
    "selected": False,
    "border_size": 20.0,
    "border_height": 1.0,
    "terrain_length": 8.0,
    "terrain_width": 8.0,
    "platform_size": 2.0,
    "num_rows": 1, # number of terrain rows (levels)
    "num_cols": 1, # number of terrain cols (types)
    "num_subterrains": 1,
    "horizontal_scale": 0.1, # [m] distance between height samples in x and y direction
    "vertical_scale": 0.005, # [m] distance between height samples in z direction
    "static_friction": 1.0, # coefficient of static friction of the terrain
    "dynamic_friction": 1.0, # coefficient of dynamic friction of the terrain
    "restitution": 0.0, # coefficient of restitution of the terrain
    "max_init_terrain_level": 1, # starting curriculum level
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, hurtle, stepping stones, gap, pit]
    "terrain_proportions": [0.0, 0.0, 1.0, 0.0, 0.0]
}

def export_png_from_height_field(height_field: np.ndarray, filename: str):
    """
    Export a height field as a PNG image.
    """
    vertical_scale = terrain_cfg["vertical_scale"]
    horizontal_scale = terrain_cfg["horizontal_scale"]
    height_field = height_field.astype(np.float32) * vertical_scale
    height_min = float(height_field.min())
    height_max = float(height_field.max())
    Zscale = max(height_max - height_min, 1e-6)
    Hn = (height_field - height_min) / Zscale + 1e-6
    image = Image.fromarray((Hn.T * 65535).astype(np.uint16), mode="I;16")
    image.save(filename)

if __name__ == "__main__":
    terrain = Terrain(terrain_cfg)
    height_field = terrain.height_field_raw
    # Save alongside this script (independent of the current working directory).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "height_field.png")
    export_png_from_height_field(height_field, out_path)
