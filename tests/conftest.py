"""pytest 配置：让测试能找到 maldi-ms-pipeline 与本项目源码。

``maldi-ms-pipeline`` 是本地依赖（``pip install -e ../maldi-ms-pipeline`` 或
设置环境变量 ``MALDI_PIPELINE_SRC`` 指向它的 ``src`` 目录）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_candidates = []
_env = os.environ.get("MALDI_PIPELINE_SRC")
if _env:
    _candidates.append(Path(_env))
_candidates.append(ROOT.parent / "maldi-ms-pipeline" / "src")

PIPELINE_SRC: Path | None = None
for _p in _candidates:
    if (_p / "maldi_ms" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        PIPELINE_SRC = _p
        break
else:
    raise RuntimeError(
        "找不到 maldi-ms-pipeline。请 pip install -e ../maldi-ms-pipeline，"
        "或设置环境变量 MALDI_PIPELINE_SRC 指向它的 src 目录。"
    )

CFG_PATH: Path | None = None
for _cand in sorted((PIPELINE_SRC.parent / "configs").glob("*.yaml")):
    CFG_PATH = _cand
    break

# ---------------------------------------------------------------- 共享 fixture

import pytest  # noqa: E402

from maldi_ms.io import load_config  # noqa: E402

from maldi_ms_agent.synth import SynthConfig, make_synth_dataset, synth_config  # noqa: E402


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    """pipeline 的默认配置，按合成谱特点调整峰检测参数。"""
    if CFG_PATH is None:
        pytest.skip("找不到 pipeline 的 configs 目录")
    return synth_config(load_config(str(CFG_PATH)))


@pytest.fixture(scope="session")
def synth_ds():
    """一份带真实信号的合成数据集（12 病例 / 12 对照）。"""
    return make_synth_dataset(SynthConfig(n_case=12, n_ctrl=12, effect_size=1.2, seed=7))
