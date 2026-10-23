# -*- coding: utf-8 -*-
"""pytest 가 없는 환경용 최소 러너. 평소에는 `pytest tests` 를 쓴다."""
import sys, traceback, inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
# 진짜 pytest 가 있으면 그쪽이 이긴다. 없을 때만 대역을 쓴다.
sys.path.append(str(ROOT / "tools" / "pytest_shim"))

import importlib
mods = ["test_labeling", "test_metrics", "test_features", "test_datasets"]
passed = failed = 0
for mn in mods:
    m = importlib.import_module(mn)
    fixtures = {n: f for n, f in vars(m).items()
                if callable(f) and getattr(f, "_is_fixture", False)}
    for name, fn in sorted(vars(m).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            sig = inspect.signature(fn)
            kw = {}
            for p in sig.parameters:
                src = getattr(m, p)
                kw[p] = src.__wrapped__() if hasattr(src, "__wrapped__") else src()
            fn(**kw)
            passed += 1
            print(f"PASS  {mn}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL  {mn}.{name}")
            traceback.print_exc(limit=2)
print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
