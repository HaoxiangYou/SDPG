## Genesis version
commit id:`7db43e4caef2b185bf691d29fc545d6480cd224d`

## Genesis Modification
Genesis will raise error when encounter nan during simulation.
We do not want to nan interrupt our training and want to handle them explicitly during environment.
To do so, we have modify the genesis code.

### 1. Nan Handling
In file `genesis/engine/solvers/rigid/rigid_solver.py`,
change the line 1066&1068 from
`gs.raise_exception("$msg")`
to
`gs.warn("$msg")`

In the same file, line 7110 and 7113 comment out the nan check as
`# is_valid &= not ti.math.isnan(e)` (NOTE, there existing other nan check, e.g. grad, which are not handling currently)

### 2. Camera dimension
In file `externals/Genesis/genesis/engine/sensors/camera.py`

change the `get_pos()` and `get_quat()` in `BatchRendererCameraWrapper`
as 

```python
def get_pos(self):
    """Get camera position (for batch renderer)."""
    n_envs = self.sensor._manager._sim._B
    if self._pos.dim() < 2:
        return self._pos.unsqueeze(0)
    else:
        return self._pos

def get_quat(self):
    """Get camera quaternion (for batch renderer)."""
    from genesis.utils.geom import T_to_trans_quat

    if self.transform.dim() < 3:
        transform = self.transform.unsqueeze(0)
    else:
        transform = self.transform
    _, quat = T_to_trans_quat(transform)
    return quat
```