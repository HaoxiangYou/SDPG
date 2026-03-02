# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from typing import Any, Dict

import numpy as np
import trimesh

from . import terrain_utils


class Terrain:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.type = cfg["mesh_type"]
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg["terrain_length"]
        self.env_width = cfg["terrain_width"]
        self.platform_size = cfg["platform_size"]
        self.proportions = [np.sum(cfg["terrain_proportions"][: i + 1]) for i in range(len(cfg["terrain_proportions"]))]
        self.cfg["num_subterrains"] = cfg["num_rows"] * cfg["num_cols"]
        self.env_origins = np.zeros((cfg["num_rows"], cfg["num_cols"], 3))
        self.width_per_env_pixels = int(self.env_width / cfg["horizontal_scale"])
        self.length_per_env_pixels = int(self.env_length / cfg["horizontal_scale"])
        # row - length, X
        # col - width,  Y
        self.border = int(cfg["border_size"] / self.cfg["horizontal_scale"])
        self.tot_cols = int(cfg["num_cols"] * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg["num_rows"] * self.length_per_env_pixels) + 2 * self.border
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.terrain_meshes = []
        if cfg["curriculum"] and cfg["selected"]:
            raise ValueError("Curriculum and selected terrain cannot be both True.")
        if cfg["curriculum"]:
            print("Generating curriculum terrain...")
            self.curiculum()
        elif cfg["selected"]:
            print("Generating selected terrain...")
            self.selected_terrain()
        else:
            print("Generating randomized terrain...")
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self._add_terrain_border()

    def randomized_terrain(self):
        for k in range(self.cfg["num_subterrains"]):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg["num_rows"], self.cfg["num_cols"]))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg["num_cols"]):  # Y
            for i in range(self.cfg["num_rows"]):  # X
                difficulty = i / self.cfg["num_rows"]  # add difficulty along X axis, row
                choice = j / self.cfg["num_cols"] + 0.001  # change terrain type along Y axis, col

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg["terrain_kwargs"].pop("type")
        for k in range(self.cfg["num_subterrains"]):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg["num_rows"], self.cfg["num_cols"]))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.length_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.cfg["vertical_scale"],
                horizontal_scale=self.cfg["horizontal_scale"],
            )

            eval(terrain_type)(terrain, **self.cfg["terrain_kwargs"], terrain_type=self.type)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.length_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg["vertical_scale"],
            horizontal_scale=self.cfg["horizontal_scale"],
        )
        slope = difficulty * 0.4
        step_height = 0.05 + 0.15 * difficulty
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 0.3 * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0] / 2:  # slope
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=self.platform_size, terrain_type=self.type
            )
        elif choice < self.proportions[1]:  # random uniform
            terrain_utils.random_uniform_terrain(
                terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2, terrain_type=self.type
            )
        elif choice < self.proportions[3]:
            if choice < self.proportions[2]:  # stairs
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=0.4,
                step_height=step_height,
                platform_size=self.platform_size,
                terrain_type=self.type,
            )
        elif choice < self.proportions[4]:  # discrete obstacles
            terrain_utils.parkour_hurtle_terrain(
                terrain,
                num_stones=2,
                platform_len=1.0,
                stone_len=0.8 + 0.3 * difficulty,
                hurtle_height_range=[0.15 + 0.1 * difficulty, 0.20 + 0.25 * difficulty],
                pad_height=0,
                x_range=[1.5, 2.5],
                y_range=[0.0, 0.0],
                half_valid_width=[0.6, 0.9],
            )

        elif choice < self.proportions[5]:  # stepping stones
            terrain_utils.stepping_stones_terrain(
                terrain,
                stone_size=stepping_stones_size,
                stone_distance=stone_distance,
                max_height=0.0,
                platform_size=self.platform_size,
                terrain_type=self.type,
            )
        elif choice < self.proportions[6]:  # gap
            terrain_utils.gap_terrain(
                terrain, gap_size=gap_size, platform_size=self.platform_size, terrain_type=self.type
            )
        else:  # pit
            terrain_utils.pit_terrain(
                terrain, depth=pit_depth, platform_size=self.platform_size, terrain_type=self.type
            )

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # env_origin_x = (i + 0.5) * self.env_length
        env_origin_x = i * self.env_length + 1.0
        env_origin_y = (j + 0.5) * self.env_width
        # use the origin height as the max height of a 2mx2m square
        # x1 = int((self.env_length / 2.0 - 0.5) / terrain.horizontal_scale)
        # x2 = int((self.env_length / 2.0 + 0.5) / terrain.horizontal_scale)
        # y1 = int((self.env_width / 2.0 - 0.5) / terrain.horizontal_scale)
        # y2 = int((self.env_width / 2.0 + 0.5) / terrain.horizontal_scale)
        # env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        # env origin z is always 0 because the terrain is always flat at the platform
        env_origin_z = 0.0
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        if self.type == "trimesh":
            # apply translation to the trimesh, align with the env origin
            translation = np.array([start_x * terrain.horizontal_scale, start_y * terrain.horizontal_scale, 0])
            terrain.terrain_mesh.apply_translation(translation)
            self.terrain_meshes.append(terrain.terrain_mesh)

    # ---------- Protected Methods ----------#

    def _add_terrain_border(self):
        """Add a surrounding border over all the sub-terrains into the terrain meshes."""
        # border parameters
        border_size = (
            self.cfg["num_rows"] * self.cfg["terrain_length"] + 2 * self.cfg["border_size"],
            self.cfg.num_cols * self.cfg.terrain_width + 2 * self.cfg.border_size,
        )
        inner_size = (
            self.cfg.num_rows * self.cfg.terrain_length
            - self.cfg.horizontal_scale,  # a small offset to align the subterrain with border
            self.cfg.num_cols * self.cfg.terrain_width - self.cfg.horizontal_scale,
        )
        border_center = (
            self.cfg.num_rows * self.cfg.terrain_length / 2 + self.cfg.border_size,
            self.cfg.num_cols * self.cfg.terrain_width / 2 + self.cfg.border_size,
            -self.cfg.border_height / 2,
        )
        # border mesh
        border_meshes = terrain_utils.make_border(
            border_size, inner_size, height=abs(self.cfg.border_height), position=border_center
        )
        border = trimesh.util.concatenate(border_meshes)
        # update the faces to have minimal triangles
        selector = ~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1)
        border.update_faces(selector)
        # add the border to the list of meshes
        self.terrain_meshes.append(border)
