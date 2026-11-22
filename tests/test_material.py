"""检查本地交换资料的案例是否互相关联。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspect_fixture import read_session


class MaterialTests(unittest.TestCase):
    def test_linked_records(self):
        d = read_session()
        rows = d["images"] + d["texts"]
        self.assertEqual({row[0]:row[1] for row in rows}, d["step_versions"])
        self.assertIn(d["duplicate"], d["texts"])
        self.assertEqual(set(d["asymmetric_trial"]["candidate_only"]), {row[0] for row in d["texts"]})
        self.assertNotIn(d["next_add"], d["matches"])
        self.assertEqual(sum(pair[0] == "i-lamp" for pair in d["matches"]), 2)
        self.assertGreater(d["load_profile"]["images"]*d["load_profile"]["texts"]*8, d["load_profile"]["scratch_bytes"])
