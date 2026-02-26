from typing import List, Tuple

import numpy as np
import trimesh
from scipy import interpolate


class SubTerrain:
    def __init__(
        self,
        terrain_name: str = "terrain",
        width: int = 256,
        length: int = 256,
        vertical_scale: float = 1.0,
        horizontal_scale: float = 1.0,
    ):
        """
        Initialize a SubTerrain object.

        Args:
            terrain_name (str, optional): Name of the terrain. Defaults to "terrain".
            width (int, optional): Width of the terrain (number of pixels). Defaults to 256.
            length (int, optional): Length of the terrain (number of pixels). Defaults to 256.
            vertical_scale (float, optional): Vertical scale of the terrain. Defaults to 1.0.
            horizontal_scale (float, optional): Horizontal scale of the terrain. Defaults to 1.0.
        """
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        # add a trimesh object to store the trimesh of subterrain, for use in trimesh terrain type
        self.terrain_mesh = trimesh.Trimesh()


# ---------- Heightfield Terrain Functions ----------#


def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float,
    max_height: float,
    step: float = 1,
    downsampled_scale: float = None,
    terrain_type: str = None,
) -> SubTerrain:
    """
    Generate a uniform noise terrain

    Parameters:
        terrain (SubTerrain): the terrain
        min_height (float): the minimum height of the terrain [meters]
        max_height (float): the maximum height of the terrain [meters]
        step (float): minimum height change between two points [meters]
        downsampled_scale (float): distance between two randomly sampled points ( musty be larger or equal to terrain.horizontal_scale)

    Note:
        This function can only generate terrain from heightfield

    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    if terrain_type in [None, "plane"]:
        raise ValueError("random_uniform_terrain can only be used for heightfield or trimesh terrain type")

    flat_edge = int(0.2 / terrain.horizontal_scale)  # 20cm flat edge around the terrain
    # switch parameters to discrete units
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(
        heights_range,
        (
            int((terrain.width - 2 * flat_edge) * terrain.horizontal_scale / downsampled_scale),
            int((terrain.length - 2 * flat_edge) * terrain.horizontal_scale / downsampled_scale),
        ),
    )

    x = np.linspace(
        flat_edge * terrain.horizontal_scale,
        (terrain.width - flat_edge) * terrain.horizontal_scale,
        height_field_downsampled.shape[0],
    )
    y = np.linspace(
        flat_edge * terrain.horizontal_scale,
        (terrain.length - flat_edge) * terrain.horizontal_scale,
        height_field_downsampled.shape[1],
    )

    # f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')
    f = interpolate.RectBivariateSpline(y, x, height_field_downsampled, kx=1, ky=1)

    x_upsampled = np.linspace(
        flat_edge * terrain.horizontal_scale,
        (terrain.width - flat_edge) * terrain.horizontal_scale,
        terrain.width - 2 * flat_edge,
    )
    y_upsampled = np.linspace(
        flat_edge * terrain.horizontal_scale,
        (terrain.length - flat_edge) * terrain.horizontal_scale,
        terrain.length - 2 * flat_edge,
    )
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw[flat_edge:-flat_edge, flat_edge:-flat_edge] += z_upsampled.astype(np.int16)

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(
            terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale
        )
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale),
            inner_size=(
                (terrain.length - 2) * terrain.horizontal_scale,
                (terrain.width - 2) * terrain.horizontal_scale,
            ),
            height=1.0,
            position=(
                0.5 * terrain.length * terrain.horizontal_scale,
                0.5 * terrain.width * terrain.horizontal_scale,
                -0.5,
            ),
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([terrain.horizontal_scale, terrain.horizontal_scale, 0])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])

    return terrain


def pyramid_sloped_terrain(
    terrain: SubTerrain, slope: float = 1, platform_size: float = 1.0, terrain_type: str = None
) -> SubTerrain:
    """
    Generate a sloped terrain

    Parameters:
        terrain (terrain): the terrain
        slope (int): positive or negative slope
        platform_size (float): size of the flat platform at the center of the terrain [meters]
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function can only generate terrain from heightfield
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("pyramid_sloped_terrain can only be used for heightfield or trimesh terrain type")

    flat_edge = int(0.2 / terrain.horizontal_scale)  # 20cm flat edge around the terrain

    x = np.arange(flat_edge, terrain.width - flat_edge)
    y = np.arange(flat_edge, terrain.length - flat_edge)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(terrain.width - 2 * flat_edge, 1)
    yy = yy.reshape(1, terrain.length - 2 * flat_edge)
    max_height = int(
        slope * (terrain.horizontal_scale / terrain.vertical_scale) * ((terrain.width - 2 * flat_edge) / 2)
    )
    terrain.height_field_raw[flat_edge:-flat_edge, flat_edge:-flat_edge] += (max_height * xx * yy).astype(
        terrain.height_field_raw.dtype
    )

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    # x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    # y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(
            terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale
        )
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale),
            inner_size=(
                (terrain.length - 2) * terrain.horizontal_scale,
                (terrain.width - 2) * terrain.horizontal_scale,
            ),
            height=1.0,
            position=(
                0.5 * terrain.length * terrain.horizontal_scale,
                0.5 * terrain.width * terrain.horizontal_scale,
                -0.5,
            ),
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([terrain.horizontal_scale, terrain.horizontal_scale, 0])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])

    return terrain


def discrete_obstacles_terrain(
    terrain: SubTerrain,
    max_height: float,
    min_size: float,
    max_size: float,
    num_rects: int,
    platform_size: float = 1.0,
    terrain_type: str = None,
) -> SubTerrain:
    """
    Generate a terrain with gaps

    Parameters:
        terrain (SubTerrain): the terrain
        max_height (float): maximum height of the obstacles (range=[-max, -max/2, max/2, max]) [meters]
        min_size (float): minimum size of a rectangle obstacle [meters]
        max_size (float): maximum size of a rectangle obstacle [meters]
        num_rects (int): number of randomly generated obstacles
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
    Returns:
        terrain (SubTerrain): update terrain
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("discrete_obstacles_terrain can only be used for heightfield or trimesh terrain type")

    # switch parameters to discrete units
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i - width, 4))
        start_j = np.random.choice(range(0, j - length, 4))
        terrain.height_field_raw[start_i : start_i + width, start_j : start_j + length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(
            terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale
        )
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale),
            inner_size=(
                (terrain.length - 2) * terrain.horizontal_scale,
                (terrain.width - 2) * terrain.horizontal_scale,
            ),
            height=1.0,
            position=(
                0.5 * terrain.length * terrain.horizontal_scale,
                0.5 * terrain.width * terrain.horizontal_scale,
                -0.5,
            ),
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([terrain.horizontal_scale, terrain.horizontal_scale, 0])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])

    return terrain


def wave_terrain(
    terrain: SubTerrain, num_waves: int = 1, amplitude: float = 1.0, terrain_type: str = None
) -> SubTerrain:
    """
    Generate a wavy terrain

    Parameters:
        terrain (SubTerrain): the terrain
        num_waves (int): number of sine waves across the terrain length
        amplitude (float): amplitude of the waves
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function can only generate terrain from heightfield
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("wave_terrain can only be used for heightfield or trimesh terrain type")

    amplitude = int(0.5 * amplitude / terrain.vertical_scale)
    flat_edge = int(0.2 / terrain.horizontal_scale)  # 20cm flat edge around the terrain
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(flat_edge, terrain.width - flat_edge)
        y = np.arange(flat_edge, terrain.length - flat_edge)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width - 2 * flat_edge, 1)
        yy = yy.reshape(1, terrain.length - 2 * flat_edge)
        terrain.height_field_raw[flat_edge : terrain.width - flat_edge, flat_edge : terrain.length - flat_edge] += (
            amplitude * np.cos(yy / div) + amplitude * np.sin(xx / div)
        ).astype(terrain.height_field_raw.dtype)

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(
            terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale
        )
        terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a border mesh to avoid holes at the edges of the terrain
        border_meshes = make_border(
            size=(terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale),
            inner_size=(
                (terrain.length - 2) * terrain.horizontal_scale,
                (terrain.width - 2) * terrain.horizontal_scale,
            ),
            height=1.0,
            position=(
                0.5 * terrain.length * terrain.horizontal_scale,
                0.5 * terrain.width * terrain.horizontal_scale,
                -0.5,
            ),
        )
        border_mesh = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border_mesh.triangles)[:, :, 2] < -0.1).any(1)
        border_mesh.update_faces(selector)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([terrain.horizontal_scale, terrain.horizontal_scale, 0])
        terrain_mesh.apply_translation(translation)
        terrain.terrain_mesh = trimesh.util.concatenate([terrain_mesh, border_mesh])

    return terrain


def pyramid_stairs_terrain(
    terrain: SubTerrain, step_width: float, step_height: float, platform_size: float = 1.0, terrain_type: str = None
) -> SubTerrain:
    """
    Generate stairs

    Parameters:
        terrain (SubTerrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float): the step_height [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        terrain_type (str): type of the terrain ("heightfield" or "trimesh")
    Returns:
        terrain (SubTerrain): update terrain
    Note:
        This function will use mesh_pyramid_stairs_terrain to directly generate the mesh
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("pyramid_stairs_terrain can only be used for heightfield or trimesh terrain type")

    # switch parameters to discrete units
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = height
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        if step_height >= 0:
            terrain.terrain_mesh = mesh_pyramid_stairs_terrain(
                terrain,
                step_width=step_width * terrain.horizontal_scale,
                step_height=step_height * terrain.vertical_scale,
                platform_size=platform_size * terrain.horizontal_scale,
            )
        elif step_height < 0:
            terrain.terrain_mesh = mesh_inverted_pyramid_stairs_terrain(
                terrain,
                step_width=step_width * terrain.horizontal_scale,
                step_height=-step_height * terrain.vertical_scale,
                platform_size=platform_size * terrain.horizontal_scale,
            )

    return terrain


def stepping_stones_terrain(
    terrain: SubTerrain,
    stone_size: float,
    stone_distance: float,
    max_height: float,
    platform_size: float = 1.0,
    depth: float = -10,
    terrain_type: str = None,
) -> SubTerrain:
    """
    Generate a stepping stones terrain

    Parameters:
        terrain (SubTerrain): the terrain
        stone_size (float): horizontal size of the stepping stones [meters]
        stone_distance (float): distance between stones (i.e size of the holes) [meters]
        max_height (float): maximum height of the stones (positive and negative) [meters]
        platform_size (float): size of the flat platform at the center of the terrain [meters]
        depth (float): depth of the holes (default=-10.) [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    if terrain_type in [None, "plane"]:
        raise ValueError("stepping_stones_terrain can only be used for heightfield or trimesh terrain type")

    # switch parameters to discrete units
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # fill first hole
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            # fill row
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # fill first hole
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            # fill column
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        vertices, triangles = convert_heightfield_to_trimesh(
            terrain.height_field_raw, terrain.horizontal_scale, terrain.vertical_scale
        )
        terrain.terrain_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        # add a small offset to align the terrain mesh with the border
        translation = np.array([terrain.horizontal_scale / 2.0, terrain.horizontal_scale / 2.0, 0])
        terrain.terrain_mesh.apply_translation(translation)

    return terrain


def gap_terrain(
    terrain: SubTerrain, gap_size: float, platform_size: float = 1.0, terrain_type: str = None
) -> SubTerrain:
    if terrain_type in [None, "plane"]:
        raise ValueError("gap_terrain can only be used for heightfield or trimesh terrain type")

    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = platform_size // 2
    x2 = x1 + gap_size
    y1 = platform_size // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        terrain.terrain_mesh = mesh_gap_terrain(
            terrain,
            gap_size=gap_size * terrain.horizontal_scale,
            platform_size=platform_size * terrain.horizontal_scale,
        )

    return terrain


def pit_terrain(
    terrain: SubTerrain,
    depth: float,
    platform_size: float = 1.0,
    terrain_type: str = None,
) -> SubTerrain:
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth

    # generate the terrain mesh for trimesh terrain type
    if terrain_type == "trimesh":
        # platform_size in heightfield is half-width (radius) in cells
        # mesh_pit_terrain expects full platform size in meters
        terrain.terrain_mesh = mesh_pit_terrain(
            terrain, depth=depth * terrain.vertical_scale, platform_size=platform_size * 2 * terrain.horizontal_scale
        )

    return terrain


def parkour_hurdle_terrain(
    terrain,
    platform_len=1.0,
    platform_height=0.0,
    num_stones=8,
    stone_len=0.3,
    x_range=(1.5, 2.4),
    y_range=(-0.4, 0.4),
    half_valid_width=(0.4, 0.8),
    hurdle_height_range=(0.2, 0.3),
    pad_width=0.1,
    pad_height=0.5,
    flat=False,
):
    goals = np.zeros((num_stones + 2, 2))
    # terrain.height_field_raw[:] = -200

    mid_y = terrain.length // 2  # length is actually y width

    dis_x_min = round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)

    # half_valid_width = round(np.random.uniform(y_range[1]+0.2, y_range[1]+1) / terrain.horizontal_scale)
    half_valid_width = round(np.random.uniform(half_valid_width[0], half_valid_width[1]) / terrain.horizontal_scale)
    hurdle_height_max = round(hurdle_height_range[1] / terrain.vertical_scale)
    hurdle_height_min = round(hurdle_height_range[0] / terrain.vertical_scale)

    platform_len = round(platform_len / terrain.horizontal_scale)
    platform_height = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len, :] = platform_height

    stone_len = round(stone_len / terrain.horizontal_scale)
    # stone_width = round(stone_width / terrain.horizontal_scale)

    # incline_height = round(incline_height / terrain.vertical_scale)
    # last_incline_height = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len
    goals[0] = [platform_len - 1, mid_y]
    # last_dis_x = dis_x
    for i in range(num_stones):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        # Allow a "fixed" y offset (e.g. y_range=[0.0, 0.0]) without crashing.
        # Note: np.random.randint(low, high) requires high > low.
        if dis_y_max == dis_y_min:
            rand_y = dis_y_min
        else:
            rand_y = np.random.randint(dis_y_min, dis_y_max)
        dis_x += rand_x
        if not flat:
            terrain.height_field_raw[dis_x - stone_len // 2 : dis_x + stone_len // 2,] = np.random.randint(
                hurdle_height_min, hurdle_height_max
            )
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2, : mid_y + rand_y - half_valid_width
            ] = 0
            terrain.height_field_raw[
                dis_x - stone_len // 2 : dis_x + stone_len // 2, mid_y + rand_y + half_valid_width :
            ] = 0
        # last_dis_x = dis_x
        goals[i + 1] = [dis_x - rand_x // 2, mid_y + rand_y]
    final_dis_x = dis_x + np.random.randint(dis_x_min, dis_x_max)
    # import ipdb; ipdb.set_trace()
    if final_dis_x > terrain.width:
        final_dis_x = terrain.width - 0.5 // terrain.horizontal_scale
    goals[-1] = [final_dis_x, mid_y]

    # terrain.goals = goals * terrain.horizontal_scale

    # pad edges
    pad_width = int(pad_width // terrain.horizontal_scale)
    pad_height = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width] = pad_height
    terrain.height_field_raw[:, -pad_width:] = pad_height
    terrain.height_field_raw[:pad_width, :] = pad_height
    terrain.height_field_raw[-pad_width:, :] = pad_height


# ---------- Trimesh Terrain Functions ----------#
# The program will use terrains directly generated by below functions when the terrain type is set to "trimesh".
# The heightfield is only used for sampling the height of the terrain at any (x,y) location.


def make_border(
    size: Tuple[float, float], inner_size: Tuple[float, float], height: float, position: Tuple[float, float, float]
) -> List[trimesh.Trimesh]:
    """Generate meshes for a rectangular border with a hole in the middle.

    .. code:: text

        +---------------------+
        |#####################|
        |##+---------------+##|
        |##|               |##|
        |##|               |##| length
        |##|               |##| (y-axis)
        |##|               |##|
        |##+---------------+##|
        |#####################|
        +---------------------+
              width (x-axis)

    Args:
        size: The length (along x) and width (along y) of the terrain (in m).
        inner_size: The inner length (along x) and width (along y) of the hole (in m).
        height: The height of the border (in m).
        position: The center of the border (in m).

    Returns:
        A list of trimesh.Trimesh objects that represent the border.
    """
    # compute thickness of the border
    thickness_x = (size[0] - inner_size[0]) / 2.0
    thickness_y = (size[1] - inner_size[1]) / 2.0
    # generate tri-meshes for the border
    # top/bottom border
    box_dims = (size[0], thickness_y, height)
    # -- top
    box_pos = (position[0], position[1] + inner_size[1] / 2.0 + thickness_y / 2.0, position[2])
    box_mesh_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- bottom
    box_pos = (position[0], position[1] - inner_size[1] / 2.0 - thickness_y / 2.0, position[2])
    box_mesh_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # left/right border
    box_dims = (thickness_x, inner_size[1], height)
    # -- left
    box_pos = (position[0] - inner_size[0] / 2.0 - thickness_x / 2.0, position[1], position[2])
    box_mesh_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # -- right
    box_pos = (position[0] + inner_size[0] / 2.0 + thickness_x / 2.0, position[1], position[2])
    box_mesh_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    # return the tri-meshes
    return [box_mesh_left, box_mesh_right, box_mesh_top, box_mesh_bottom]


def mesh_pyramid_stairs_terrain(
    terrain: SubTerrain, step_width: float, step_height: float, platform_size: float = 1.0
) -> trimesh.Trimesh:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.


    Args:
        step_width (float): width of each step in the stairs (in m)
        step_height (float): height of each step in the stairs (in m)
        platform_size (float): size of the flat platform at the center of the terrain (in

    Returns:
        the tri-mesh of the terrain.
    """

    # compute number of steps in x and y direction
    num_steps_x = (terrain.length * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    num_steps_y = (terrain.width * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))

    # initialize list of meshes
    meshes_list = list()

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [
        0.5 * terrain.length * terrain.horizontal_scale,
        0.5 * terrain.width * terrain.horizontal_scale,
        0.0,
    ]
    terrain_size = (terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale)
    # -- generate the stair pattern
    for k in range(num_steps):
        box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] + k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = k * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]

    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        (num_steps - 1) * step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] + (num_steps - 1) * step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh


def mesh_inverted_pyramid_stairs_terrain(
    terrain: SubTerrain, step_width: float, step_height: float, platform_size: float = 1.0
) -> trimesh.Trimesh:
    """Generate a terrain with a inverted pyramid stair pattern.

    The terrain is an inverted pyramid stair pattern which trims to a flat platform at the center of the terrain.

    Args:
        terrain: The terrain configuration.
        step_width: The width of each step.
        step_height: The height of each step.
        platform_size: The size of the flat platform at the center of the terrain.

    Returns:
        The tri-mesh of the terrain.
    """
    # compute number of steps in x and y direction
    num_steps_x = (terrain.length * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    num_steps_y = (terrain.width * terrain.horizontal_scale - platform_size) // (2 * step_width) + 1
    # we take the minimum number of steps in x and y direction
    num_steps = int(min(num_steps_x, num_steps_y))
    # total height of the terrain
    total_height = (num_steps - 1) * step_height

    # initialize list of meshes
    meshes_list = list()

    # generate the terrain
    # -- compute the position of the center of the terrain
    terrain_center = [
        0.5 * terrain.length * terrain.horizontal_scale,
        0.5 * terrain.width * terrain.horizontal_scale,
        0.0,
    ]
    terrain_size = (terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale)
    # -- generate the stair pattern
    for k in range(num_steps):
        box_size = (terrain_size[0] - 2 * k * step_width, terrain_size[1] - 2 * k * step_width)
        # compute the quantities of the box
        # -- location
        box_z = terrain_center[2] - total_height / 2 - k * step_height / 2.0
        box_offset = (k + 0.5) * step_width
        # -- dimensions
        box_height = total_height - k * step_height
        # generate the boxes
        # top/bottom
        box_dims = (box_size[0], step_width, box_height)
        # -- top
        box_pos = (terrain_center[0], terrain_center[1] + terrain_size[1] / 2.0 - box_offset, box_z)
        box_top = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- bottom
        box_pos = (terrain_center[0], terrain_center[1] - terrain_size[1] / 2.0 + box_offset, box_z)
        box_bottom = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # right/left
        box_dims = (step_width, box_size[1] - 2 * step_width, box_height)
        # -- right
        box_pos = (terrain_center[0] + terrain_size[0] / 2.0 - box_offset, terrain_center[1], box_z)
        box_right = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # -- left
        box_pos = (terrain_center[0] - terrain_size[0] / 2.0 + box_offset, terrain_center[1], box_z)
        box_left = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        # add the boxes to the list of meshes
        meshes_list += [box_top, box_bottom, box_right, box_left]
    # generate final box for the middle of the terrain
    box_dims = (
        terrain_size[0] - 2 * num_steps * step_width,
        terrain_size[1] - 2 * num_steps * step_width,
        step_height,
    )
    box_pos = (terrain_center[0], terrain_center[1], terrain_center[2] - total_height - step_height / 2)
    box_middle = trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
    meshes_list.append(box_middle)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh


def mesh_gap_terrain(terrain: SubTerrain, gap_size: float, platform_size: float = 1.0) -> trimesh.Trimesh:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    Args:
        terrain (SubTerrain): The terrain configuration.
        gap_size (float): The width of the gap around the platform (in m).
        platform_size (float): The size of the flat platform at the center of the terrain (in m).

    Returns:
        Trimesh: The tri-mesh of the terrain.
    """
    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = 1.0
    terrain_center = (
        0.5 * terrain.length * terrain.horizontal_scale,
        0.5 * terrain.width * terrain.horizontal_scale,
        -terrain_height / 2,
    )
    terrain_size = (terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale)

    # Generate the outer ring
    inner_size = (platform_size + 2 * gap_size, platform_size + 2 * gap_size)
    meshes_list += make_border(terrain_size, inner_size, terrain_height, terrain_center)
    # Generate the inner box
    box_dim = (platform_size, platform_size, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh


def mesh_pit_terrain(terrain: SubTerrain, depth: float, platform_size: float = 1.0) -> trimesh.Trimesh:
    """Generate a terrain with a pit with levels (stairs) leading out of the pit.

    The terrain contains a platform at the center and a staircase leading out of the pit.
    The staircase is a series of steps that are aligned along the x- and y- axis. The steps are
    created by extruding a ring along the x- and y- axis. If :obj:`is_double_pit` is True, the pit
    contains two levels.

    Args:
        terrain (SubTerrain): The terrain configuration.
        depth (float): The depth of the pit (in m).
        platform_size (float): The size of the flat platform at the center of the terrain (in m).

    Returns:
        Trimesh: The tri-mesh of the terrain.
    """
    # resolve the terrain configuration
    pit_depth = depth

    # initialize list of meshes
    meshes_list = list()
    # extract quantities
    inner_pit_size = (platform_size, platform_size)
    total_depth = pit_depth
    # constants for terrain generation
    terrain_height = 1.0

    # generate the pit (outer ring)
    pit_center = [
        0.5 * terrain.length * terrain.horizontal_scale,
        0.5 * terrain.width * terrain.horizontal_scale,
        -total_depth * 0.5,
    ]
    terrain_size = (terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale)
    meshes_list += make_border(terrain_size, inner_pit_size, total_depth, pit_center)
    # generate the ground
    dim = (terrain.length * terrain.horizontal_scale, terrain.width * terrain.horizontal_scale, terrain_height)
    pos = (
        0.5 * terrain.length * terrain.horizontal_scale,
        0.5 * terrain.width * terrain.horizontal_scale,
        -total_depth - terrain_height / 2,
    )
    ground_meshes = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground_meshes)
    mesh = trimesh.util.concatenate(meshes_list)

    return mesh


# ---------- Utility Functions ----------#


def convert_heightfield_to_trimesh(
    height_field_raw: np.ndarray, horizontal_scale: float, vertical_scale: float, slope_threshold=None
):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return vertices, triangles
