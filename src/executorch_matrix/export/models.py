"""Model specifications for the comparison matrix.

A ``ModelSpec`` knows how to build a fresh ``nn.Module`` plus a matching tuple of
example inputs — that's all the export pipeline needs. We ship one small,
fast-to-export built-in model so the quickstart and CI run without downloading a
large LLM. Larger models (e.g. Llama 3.2 1B) are exported through ExecuTorch's
own ``examples/models`` flow and are out of scope for the bundled runnable
example; this registry is the extension point for adding more.

``torch`` is imported lazily inside the builders, so importing this module (e.g.
to look up a spec name) stays cheap and free of the heavy ML stack.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """A named model plus a builder for (module, example_inputs)."""

    name: str
    description: str
    builder: Callable[[], tuple[Any, tuple[Any, ...]]]

    def build(self) -> tuple[Any, tuple[Any, ...]]:
        """Return a fresh, eval-mode model and a matching example-input tuple."""
        model, example_inputs = self.builder()
        return model.eval(), example_inputs


def _build_tiny() -> tuple[Any, tuple[Any, ...]]:
    """A deliberately small conv classifier.

    Chosen so that (a) it exports in well under a second, (b) it exercises real
    convolution + linear ops so delegation and quantization actually do
    something, and (c) both ``Linear`` layers have ``in_features == 128`` so a
    single int4 group size divides them cleanly and every quant variant is valid.
    """
    import torch
    from torch import nn

    width = 128

    class TinyClassifier(nn.Module):
        def __init__(self, num_classes: int = 32) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, width, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),  # -> `width` features
            )
            self.classifier = nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Linear(width, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return TinyClassifier(), (torch.randn(1, 3, 32, 32),)


BUILTIN_MODELS: dict[str, ModelSpec] = {
    "tiny": ModelSpec(
        name="tiny",
        description="Small conv classifier; fast to export, used by CI and the quickstart.",
        builder=_build_tiny,
    ),
}


def resolve_model(model: str) -> ModelSpec:
    """Resolve a model name or a path to a spec JSON into a ``ModelSpec``.

    A plain name (e.g. ``"tiny"``) resolves against the built-in registry. A path
    to a JSON file is read as ``{"model": "<builtin-name>", ...}`` — the example
    config format under ``examples/sample-model``.
    """
    candidate = Path(model)
    if candidate.suffix == ".json" and candidate.exists():
        data = json.loads(candidate.read_text())
        builtin = data.get("model", "tiny")
        if builtin not in BUILTIN_MODELS:
            raise KeyError(
                f"Config {model!r} references unknown built-in model {builtin!r}. "
                f"Available: {', '.join(sorted(BUILTIN_MODELS))}."
            )
        return BUILTIN_MODELS[builtin]

    if model in BUILTIN_MODELS:
        return BUILTIN_MODELS[model]

    raise KeyError(
        f"Unknown model {model!r}. Built-in models: {', '.join(sorted(BUILTIN_MODELS))}. "
        "Pass a path to an examples/sample-model-style config.json to use your own."
    )
