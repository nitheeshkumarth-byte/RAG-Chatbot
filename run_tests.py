"""Run both assert suites in order, then restore what they touched.

Usage: python run_tests.py

test_experiments.py runs offline (no model downloads, no LLM calls).
test_api.py uploads files and rewrites index.pkl, so both are backed up first
and restored afterwards. No pytest dependency: the suites are plain scripts.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ("test_experiments.py", "test_api.py")
TOUCHED = ("index.pkl", "sample_docs")


def main() -> int:
    backup = tempfile.mkdtemp(prefix="rag_test_backup_")
    existed = {}
    try:
        for name in TOUCHED:
            src = os.path.join(ROOT, name)
            existed[name] = os.path.exists(src)
            if not existed[name]:
                continue
            dst = os.path.join(backup, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        failed = False
        for script in SCRIPTS:
            print(f"\n=== {script} ===")
            if subprocess.call([sys.executable, script], cwd=ROOT) != 0:
                failed = True
        return 1 if failed else 0
    finally:
        for name in TOUCHED:
            dst = os.path.join(ROOT, name)
            src = os.path.join(backup, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.isfile(dst):
                os.remove(dst)
            if existed[name]:
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
