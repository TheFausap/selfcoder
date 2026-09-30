import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from selfcoder.cli import build_parser, cmd_improve
from selfcoder.config import Config
from selfcoder.embeddings import HashingEmbedder
from selfcoder.memory import MemoryStore, index_codebase
from selfcoder.patcher import Edit


class ImproveTests(unittest.TestCase):
    def test_apply_reindexes_only_touched_readable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "example.py"
            source.write_text("value = 1\n")
            (root / "untouched.py").write_text("value = 10\n")
            store = MemoryStore(root / ".selfcoder/memory.db", HashingEmbedder())
            edits = [Edit("example.py", "patch", find="value = 1", replace="value = 2"),
                     Edit("new.py", "write", content="new = True\n"),
                     Edit("asset.svg", "write", content="<svg/>\n")]
            args = build_parser().parse_args(["improve", "Update value", "--yes"])
            with patch("selfcoder.cli.open_store", return_value=store), \
                 patch("selfcoder.cli.propose", return_value=({"summary": "Update"}, edits)), \
                 patch("selfcoder.cli.remember_edit", return_value=None), \
                 patch("selfcoder.cli.index_codebase", wraps=index_codebase) as index, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_improve(args, Config(), root), 0)
                self.assertEqual(index.call_args.args[1], {
                    "example.py": "value = 2\n", "new.py": "new = True\n",
                })
            self.assertEqual(source.read_text(), "value = 2\n")

    def test_dry_run_and_apply_request_the_same_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requests = []
            for dry_run in (True, False):
                store = Mock()
                args = build_parser().parse_args(
                    ["improve", "Update value", "--no-remember"] +
                    (["--dry-run"] if dry_run else [])
                )
                with patch("selfcoder.cli.open_store", return_value=store), \
                     patch("selfcoder.cli.propose", return_value=({}, [])) as propose, \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cmd_improve(args, Config(), root), 0)
                    requests.append((propose.call_args.args[2:], propose.call_args.kwargs))
            self.assertEqual(requests[0], requests[1])
