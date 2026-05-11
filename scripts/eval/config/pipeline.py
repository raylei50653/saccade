from __future__ import annotations

import dataclasses

from .core import CoreConfig
from .detection import DetectionConfig
from .geometry import GeometryConfig
from .reid import ReIDConfig
from .semantic import SemanticConfig
from .trigger import TriggerConfig
from .lifecycle import LifecycleConfig


@dataclasses.dataclass
class PipelineConfig:
    """Composes all module configs into a flat dict for run_eval."""

    core: CoreConfig = dataclasses.field(default_factory=CoreConfig)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    geometry: GeometryConfig = dataclasses.field(default_factory=GeometryConfig)
    reid: ReIDConfig = dataclasses.field(default_factory=ReIDConfig)
    semantic: SemanticConfig = dataclasses.field(default_factory=SemanticConfig)
    trigger: TriggerConfig = dataclasses.field(default_factory=TriggerConfig)
    lifecycle: LifecycleConfig = dataclasses.field(default_factory=LifecycleConfig)

    def to_flat_dict(self) -> dict:
        result: dict = {}
        for mod in (
            self.core,
            self.detection,
            self.geometry,
            self.reid,
            self.semantic,
            self.trigger,
            self.lifecycle,
        ):
            result.update(mod.to_flat_dict())
        return result

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Build PipelineConfig from parsed argparse namespace.

        Each module is populated directly from args attributes, so CLI overrides
        always win over module YAML defaults (which were applied via set_defaults).
        """
        d = vars(args)
        return cls(
            core=_pick(CoreConfig, d),
            detection=_pick(DetectionConfig, d),
            geometry=_pick(GeometryConfig, d),
            reid=_pick(ReIDConfig, d),
            semantic=_pick(SemanticConfig, d),
            trigger=_pick(TriggerConfig, d),
            lifecycle=_pick(LifecycleConfig, d),
        )


def _pick(cls, d: dict):
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})
