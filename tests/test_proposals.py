import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from selfcoder.agent import propose
from selfcoder.embeddings import HashingEmbedder
from selfcoder.memory import MemoryStore, index_codebase
from selfcoder.patcher import PatchError, Patcher


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "example.py"
        self.original = "def verify():\n    return 'current'\n"
        self.source.write_text(self.original)
        self.store = MemoryStore(self.root / ".selfcoder/memory.db", HashingEmbedder())
        self.addCleanup(self.store.close)
        index_codebase(self.store, {"example.py": "def verify():\n    return 'stale'\n"})

    def plan(self, find):
        return {"edits": [{"file": "example.py", "action": "patch",
                           "find": find, "replace": "    return 'updated'"}]}

    def propose(self, client):
        return propose(client, self.store, "Update verify", k=4, budget=10000,
                       root=self.root)

    def test_current_source_replaces_stale_index(self):
        client = Mock()
        client.chat_json.return_value = self.plan("    return 'current'")
        _, edits = self.propose(client)
        prompt = client.chat_json.call_args.args[0][1]["content"]
        self.assertIn(self.original, prompt)
        self.assertNotIn("return 'stale'", prompt)
        self.assertIn("+    return 'updated'", Patcher(self.root).preview(edits).diff)
        self.assertEqual(self.source.read_text(), self.original)

    def test_invalid_snippet_gets_one_repair_without_writes(self):
        client = Mock()
        client.chat_json.side_effect = [self.plan("invented snippet"),
                                        self.plan("    return 'current'")]
        _, edits = self.propose(client)
        self.assertEqual(client.chat_json.call_count, 2)
        repair = client.chat_json.call_args.args[0][-1]["content"]
        self.assertIn("snippet was not found", repair)
        self.assertIn(self.original, repair)
        self.assertIn("updated", Patcher(self.root).preview(edits).diff)
        self.assertEqual(self.source.read_text(), self.original)
        self.assertFalse((self.root / ".selfcoder/backups").exists())

    def test_named_function_is_included_when_vector_search_misses(self):
        client = Mock()
        client.chat_json.return_value = self.plan("    return 'current'")
        store = Mock()
        store.search.return_value = []
        propose(client, store, "Update verify", k=4, budget=10000, root=self.root)
        prompt = client.chat_json.call_args.args[0][1]["content"]
        self.assertIn(self.original, prompt)
        self.assertEqual(self.source.read_text(), self.original)

    def test_repeated_invalid_proposal_stops(self):
        client = Mock()
        client.chat_json.return_value = self.plan("invented snippet")
        with self.assertRaises(PatchError):
            self.propose(client)
        self.assertEqual(client.chat_json.call_count, 2)
        self.assertEqual(self.source.read_text(), self.original)

    def test_existing_file_rewrite_is_rejected_and_repaired(self):
        client = Mock()
        client.chat_json.side_effect = [
            {"edits": [{"file": "example.py", "action": "write", "content": ""}]},
            self.plan("    return 'current'"),
        ]
        _, edits = self.propose(client)
        self.assertEqual(client.chat_json.call_count, 2)
        self.assertIn("whole-file replacement", client.chat_json.call_args.args[0][-1]["content"])
        self.assertEqual(edits[0].action, "patch")
        self.assertEqual(self.source.read_text(), self.original)

    def test_new_file_can_still_be_created(self):
        client = Mock()
        client.chat_json.return_value = {
            "edits": [{"file": "new.py", "action": "write", "content": "value = 1\n"}]
        }
        _, edits = self.propose(client)
        self.assertEqual(Patcher(self.root).preview(edits).created, ["new.py"])
        self.assertFalse((self.root / "new.py").exists())

    def test_other_goal_memories_are_excluded(self):
        self.store.add("edit", "UNRELATED_TASK: add side-effect analysis rules",
                       source="Add side-effect analysis rules", label="earlier edit")
        self.store.add("lesson", "UNRELATED_RETRY: fix escaped newlines",
                       source="Add side-effect analysis rules", label="earlier retry")
        self.store.add("analysis", "UNRELATED_REVIEW: change ANALYZE_SYSTEM",
                       source="Update verify", label="review")
        client = Mock()
        client.chat_json.return_value = self.plan("    return 'current'")
        self.propose(client)
        prompt = client.chat_json.call_args.args[0][1]["content"]
        self.assertNotIn("UNRELATED_TASK", prompt)
        self.assertNotIn("UNRELATED_RETRY", prompt)
        self.assertNotIn("UNRELATED_REVIEW", prompt)
        self.assertIn("Current goal: Update verify", prompt)

    def test_same_goal_history_is_retained(self):
        self.store.add("lesson", "SAME_GOAL_LESSON: preserve return type",
                       source="  UPDATE   verify  ", label="verification lesson")
        client = Mock()
        client.chat_json.return_value = self.plan("    return 'current'")
        self.propose(client)
        prompt = client.chat_json.call_args.args[0][1]["content"]
        self.assertIn("SAME_GOAL_LESSON", prompt)

    def test_syntax_error_gets_repaired_before_preview(self):
        client = Mock()
        invalid = self.plan("    return 'current'")
        invalid["edits"][0]["replace"] = (
            "    return 'updated'\n\n9. Trace each proposed edit.\n"
        )
        client.chat_json.side_effect = [invalid, self.plan("    return 'current'")]
        _, edits = self.propose(client)
        repair = client.chat_json.call_args.args[0][-1]["content"]
        self.assertIn("syntax error", repair)
        self.assertIn("Current goal: Update verify", repair)
        self.assertIn("Discard any changes addressing other goals", repair)
        Patcher(self.root).preview(edits, check_syntax=True)
        self.assertEqual(self.source.read_text(), self.original)

    def test_repeated_syntax_error_is_rejected_without_writes(self):
        client = Mock()
        invalid = self.plan("    return 'current'")
        invalid["edits"][0]["replace"] = "    return ("
        client.chat_json.return_value = invalid
        with self.assertRaisesRegex(PatchError, "syntax error"):
            self.propose(client)
        self.assertEqual(client.chat_json.call_count, 2)
        self.assertEqual(self.source.read_text(), self.original)


if __name__ == "__main__":
    unittest.main()
