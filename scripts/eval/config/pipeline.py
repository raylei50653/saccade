from __future__ import annotations

import dataclasses

from .core import CoreConfig
from .detection import DetectionConfig
from .geometry import GeometryConfig
from .motion import MotionConfig
from .reid import ReIDConfig
from .semantic import SemanticConfig
from .trigger import TriggerConfig
from .lifecycle import LifecycleConfig


@dataclasses.dataclass
class PipelineConfig:
    """Typed *partial* view over the module configs.

    WARNING — this is NOT the runtime config path. ``scripts/eval/mot17.py``
    feeds ``vars(args)`` (the raw argparse namespace) straight into ``run_eval``;
    nothing in the runtime calls ``from_args`` / ``to_flat_dict``.

    It is also INCOMPLETE: a number of tracking knobs are declared only as
    argparse arguments (in the ``add_*_args`` functions) and have no
    corresponding dataclass field, so ``to_flat_dict`` silently omits them —
    e.g. ``occ_*``, ``oao_*``, ``sinkhorn_lambda``, ``multiplicative_cost``,
    ``stability_cost_w``, ``multi_birth_replace_*``. Do not treat the result as
    a faithful snapshot of a run; use ``scripts/eval/print_assoc_basis.py``
    (which resolves the argparse namespace) for that.
    """

    core: CoreConfig = dataclasses.field(default_factory=CoreConfig)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    geometry: GeometryConfig = dataclasses.field(default_factory=GeometryConfig)
    motion: MotionConfig = dataclasses.field(default_factory=MotionConfig)
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
            self.motion,
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
            motion=_pick(MotionConfig, d),
            reid=_pick(ReIDConfig, d),
            semantic=_pick(SemanticConfig, d),
            trigger=_pick(TriggerConfig, d),
            lifecycle=_pick(LifecycleConfig, d),
        )


def _pick(cls, d: dict):
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})
