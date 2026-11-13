"""评测与数据治理流水线的端到端测试。

这些测试不是"函数返回了什么"，而是**结论与门禁的行为**：同一份冻结报告，
在 CI 和本地必须得到同样的 p 值、同样的拦截方向、同样的审计留痕。

平台（``maldi-ms-platform``）是可选依赖：矩阵里没 checkout 它就跳过，
但 CI 有一个专门的 job 会带着它把这几个测试跑一遍——跳过不等于放过。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import governance  # noqa: E402  （平台只在函数内 import，模块本身可安全导入）

try:
    governance._ensure_platform()
except RuntimeError as exc:  # 矩阵里没 checkout 平台时会走到这里
    pytest.skip(str(exc), allow_module_level=True)

pytest.importorskip("maldi_platform")


@pytest.fixture(scope="module")
def report(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("gov") / "governance.json"
    rc = governance.main(["--out", str(out)])
    assert rc == 0, "治理流水线的断言没全过，先看上面打印的未通过项"
    return json.loads(out.read_text(encoding="utf-8"))


def test_reproduces_published_numbers(report: dict) -> None:
    """四个臂的通过数与公开口径逐格一致：11/16、11/16、0/16、11/16。"""
    arms = {r["arm"]: (r["k"], r["n"]) for r in report["runs"]}
    assert arms == {
        "A_full": (11, 16),
        "B_no_warning": (11, 16),
        "C_no_tool": (0, 16),
        "D_loose_prompt": (11, 16),
    }


def test_paired_test_not_rate_comparison(report: dict) -> None:
    """比较用的是配对 McNemar 精确检验，不是"两个比率谁大"。"""
    cmp = report["compare"]
    assert cmp["table"]["a_only"] == 11
    assert cmp["table"]["b_only"] == 0
    assert cmp["mcnemar_p"] < 0.001
    assert cmp["verdict"] == "差异显著"


def test_gate_blocks_regression_but_not_reverse(report: dict) -> None:
    """门禁有方向：候选显著变差要拦，反向要放——否则只是"永远拒绝"。"""
    gates = report["gates"]
    assert gates["no_regression_candidate_vs_baseline"]["passed"] is False
    assert gates["no_regression_baseline_vs_candidate"]["passed"] is True


def test_tool_gate_catches_missing_permutation(report: dict) -> None:
    """过程性门禁：C 组没调用置换检验，即使不看结果也该被拦。"""
    gates = report["gates"]
    assert gates["require_tool_candidate"]["passed"] is False
    assert gates["require_tool_baseline"]["passed"] is True


def test_real_data_payload_export_denied_and_audited(report: dict) -> None:
    """真实受试者数据只能出元数据；被拒这件事本身要留痕。"""
    real = report["dataset"]["real"]
    assert real["classification"] == "real"
    assert real["payload_export"] == "denied"
    assert "拒绝" in real["denied_reason"]
    assert real["content_hash"].startswith("external:")  # 只登记，不复制数据
    assert report["audit"]["denied"] >= 1


def test_audit_log_is_append_only(report: dict) -> None:
    assert report["audit"]["append_only"] is True
    assert report["audit"]["n_entries"] >= 4


def test_dataset_version_is_content_addressed(report: dict) -> None:
    """数据版本身份来自内容哈希，不是"第几版"这种自报编号。"""
    corpus = report["dataset"]["corpus"]
    h = corpus["content_hash"]
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
    assert corpus["n_cases"] == 16


def test_tampered_report_is_refused(tmp_path, monkeypatch) -> None:
    """冻结产物被改过就拒绝启动——不能让一份来路不明的报告继续产生结论。"""
    manifest = json.loads(governance.MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = "0" * 64
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(governance, "MANIFEST_PATH", tampered)
    assert governance.main(["--out", str(tmp_path / "x.json")]) == 2


def test_crlf_artifact_hashes_same_as_lf(tmp_path) -> None:
    """同一个文件在 CRLF 与 LF 下必须得到同一个身份。

    踩过的坑：本地 Windows 检出是 CRLF、CI 是 LF，按原始字节算哈希会让
    "本地全绿、CI 退出码 2"——内容寻址最典型的跨平台事故。
    """
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{"k": 1}\n{"k": 2}\n')
    crlf.write_bytes(b'{"k": 1}\r\n{"k": 2}\r\n')
    assert governance._sha256(lf) == governance._sha256(crlf)

    # 真实产物同样要过：报告里有 60 万字节，足以覆盖分块读取的边界情况
    report_crlf = tmp_path / "report_crlf.json"
    report_crlf.write_bytes(governance.REPORT_PATH.read_bytes().replace(b"\n", b"\r\n"))
    assert governance._sha256(report_crlf) == governance._sha256(governance.REPORT_PATH)
