#!/usr/bin/env python3
"""启动 maldi-ms-agent 的 MCP stdio 工具服务。

默认绑定合成数据集（真实 37 例谱不出域、不随仓分发），
可用 ``--allowed`` 覆盖默认工具白名单。

    python scripts/run_mcp_server.py
    python scripts/run_mcp_server.py --once '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
    echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"finish","arguments":{"conclusion":"证据不足"}}}' \\
        | python scripts/run_mcp_server.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# maldi-ms-pipeline 是本地依赖：CI 用 MALDI_PIPELINE_SRC 指定，
# 本地默认找同级目录；与 tests/conftest.py 的查找顺序保持一致。
_candidates = []
if os.environ.get("MALDI_PIPELINE_SRC"):
    _candidates.append(Path(os.environ["MALDI_PIPELINE_SRC"]))
_candidates.append(ROOT.parent / "maldi-ms-pipeline" / "src")
for _p in _candidates:
    if (_p / "maldi_ms" / "__init__.py").exists():
        sys.path.insert(0, str(_p))
        break

from maldi_ms_agent.mcp_server import build_mcp_server  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="maldi-ms-agent MCP stdio server")
    ap.add_argument("--once", default="", help="只处理一帧 JSON 请求后退出（调试用）")
    ap.add_argument("--allowed", default="", help="逗号分隔的工具白名单（默认排除 load_dataset）")
    ap.add_argument("--max-sessions", type=int, default=8, help="会话 LRU 上限")
    args = ap.parse_args(argv)

    allowed = [x.strip() for x in args.allowed.split(",") if x.strip()] or None
    server = build_mcp_server(allowed_tools=allowed, max_sessions=args.max_sessions)

    if args.once:
        reply = server.handle(args.once)
        if reply is not None:
            print(reply)
        return 0

    print(
        "[mcp] maldi-ms-agent stdio server ready; "
        "synthetic data bound, load_dataset not exposed by default",
        file=sys.stderr,
    )
    server.serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
