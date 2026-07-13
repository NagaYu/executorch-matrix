"""executorch-matrix: compare ExecuTorch backends and quantization levels.

Orchestrates ExecuTorch's own export, quantization (torchao), and profiling
(ETDump/ETRecord) tooling across backend x quantization combinations, then
produces a comparison report and a reasoned recommendation. It does not
reimplement any of ExecuTorch's underlying capabilities.
"""

__version__ = "0.1.0"
