import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from selfcoder.config import Config


class CodingConfigTests(unittest.TestCase):
    def test_fallback_and_independent_overrides(self):
        base = Config(base_url='http://analysis/v1', model='analysis')
        self.assertEqual(base.for_coding().base_url, base.base_url)
        self.assertEqual(base.for_coding().model, base.model)
        base.coding_url = 'http://coding/v1'
        self.assertEqual(base.for_coding().base_url, base.coding_url)
        self.assertEqual(base.for_coding().model, 'analysis')
        base.coding_model = 'coder'
        self.assertEqual(base.for_coding().model, 'coder')
        self.assertEqual(base.model, 'analysis')
        self.assertEqual(base.base_url, 'http://analysis/v1')
        base.coding_url = None
        self.assertEqual(base.for_coding().base_url, base.base_url)

    def test_file_and_environment_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps({'coding_url': 'http://coding/v1', 'coding_model': 'v1'}))
            with patch.dict(os.environ, {}, clear=True):
                config = Config.load(path)
                self.assertEqual(config.coding_model, 'v1')
                self.assertEqual(config.coding_url, 'http://coding/v1')
            with patch.dict(os.environ, {'SELFCODER_CODING_URL': 'http://override/v1',
                                         'SELFCODER_CODING_MODEL': 'override'}, clear=True):
                config = Config.load(path)
                self.assertEqual(config.for_coding().base_url, 'http://override/v1')
                self.assertEqual(config.for_coding().model, 'override')
