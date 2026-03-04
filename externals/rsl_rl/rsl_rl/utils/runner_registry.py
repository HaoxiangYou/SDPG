# remove: from rsl_rl.runners import OnPolicyRunner
from typing import Any, Type


class RunnerRegistry:
    def __init__(self):
        self.runner_classes: dict[str, Type[Any]] = {}

    def register(self, name: str, runner_class: Type[Any]) -> None:
        self.runner_classes[name] = runner_class

    def get_runner_class(self, name: str) -> Type[Any]:
        return self.runner_classes[name]


runner_registry = RunnerRegistry()
