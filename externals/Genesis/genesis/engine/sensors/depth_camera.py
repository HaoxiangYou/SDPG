import torch

from genesis.options.sensors import DepthCamera as DepthCameraOptions

from .raycaster import RaycasterData, RaycasterSensor, RaycasterSharedMetadata
from .sensor_manager import register_sensor


@register_sensor(DepthCameraOptions, RaycasterSharedMetadata, RaycasterData)
class DepthCameraSensor(RaycasterSensor):
    def build(self):
        super().build()
        env_idx = self._shared_metadata.env_idx
        if env_idx is not None:
            batch_size = len(env_idx)
        else:
            batch_size = self._manager._sim._B
        batch_shape = (batch_size,) if self._manager._sim.n_envs > 0 else ()
        self._shape = (*batch_shape, self._options.pattern.height, self._options.pattern.width)

    def read_image(self, envs_idx=None) -> torch.Tensor:
        """
        Read the depth image from the sensor.

        This method uses the hit distances from the underlying RaycasterSensor.read() method and reshapes into image.

        Parameters
        ----------
        envs_idx : array_like or None
            Real environment indices to read.  When ``env_idx`` subset rendering
            is active, these are transparently mapped to compact cache rows.
            If *None*, all rendered envs are returned (compact when ``env_idx``
            is set, otherwise all envs).

        Returns
        -------
        torch.Tensor
            The depth image with shape ``(n, height, width)`` (batched) or
            ``(height, width)`` (single env / non-batched scene).
        """
        distances = self.read(envs_idx=envs_idx).distances
        h, w = self._options.pattern.height, self._options.pattern.width
        if self._manager._sim.n_envs > 0:
            return distances.reshape(-1, h, w)
        return distances.reshape(h, w)
