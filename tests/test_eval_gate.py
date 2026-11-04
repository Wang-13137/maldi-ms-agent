"""行为门禁的规则层测试。

完整门禁（8 条 check + 注入自检）在 CI 里由 ``scripts/eval_gate.py`` 单独跑；
这里只测单条规则，避免 CI 重复造数据。核心是验证**规则真的有区分力**：
好 Agent 过、故意虚报的坏 Agent 必须被挡。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eval_gate  # noqa: E402

from maldi_ms_agent.llm import ScriptedLLM  # noqa: E402
from maldi_ms_agent.rule_agent import RuleBasedLLM  # noqa: E402
from maldi_ms_agent.tasks import TASKS, run_task  # noqa: E402


def _trial(kind: str, llm, base_cfg):
    task = next(t for t in TASKS if t.kind == kind)
    return run_task(task, base_cfg, llm, max_steps=14)


def _good(base_cfg):
    return _trial("null", RuleBasedLLM(n_perm=200), base_cfg)


def _bad(base_cfg):
    return _trial("null", ScriptedLLM([eval_gate.BAD_CLAIM]), base_cfg)


def test_no_false_claim_distinguishes_good_and_bad(base_cfg):
    assert eval_gate._no_false_claim([_good(base_cfg)])[0] is True
    assert eval_gate._no_false_claim([_bad(base_cfg)])[0] is False


def test_permutation_hard_constraint_catches_shortcut(base_cfg):
    assert eval_gate._permutation_before_conclusion([_good(base_cfg)])[0] is True
    assert eval_gate._permutation_before_conclusion([_bad(base_cfg)])[0] is False


def test_step_budget_rule(base_cfg):
    good = [_good(base_cfg)]
    assert eval_gate._step_budget(good, 14)[0] is True
    assert eval_gate._step_budget(good, 0)[0] is False


def test_trace_and_redaction_and_otlp_rules(base_cfg):
    good = [_good(base_cfg)]
    assert eval_gate._trace_complete(good)[0] is True
    assert eval_gate._default_redaction(good)[0] is True
    assert eval_gate._otlp_exportable(good)[0] is True
    assert eval_gate._stable_eval_schema(good)[0] is True


def test_main_writes_artifact_and_returns_exit_code(tmp_path, monkeypatch):
    out = tmp_path / "reports" / "gate.json"
    fake_pass = {
        "checks": [{"name": "x", "passed": True, "detail": "ok"}],
        "n_passed": 1,
        "n_checks": 1,
        "passed": True,
    }
    monkeypatch.setattr(eval_gate, "run_gate", lambda **kw: fake_pass)
    assert eval_gate.main(["--out", str(out)]) == 0
    assert out.exists()

    fake_fail = {
        "checks": [{"name": "x", "passed": False, "detail": "bad"}],
        "n_passed": 0,
        "n_checks": 1,
        "passed": False,
    }
    monkeypatch.setattr(eval_gate, "run_gate", lambda **kw: fake_fail)
    assert eval_gate.main(["--out", str(out)]) == 1
