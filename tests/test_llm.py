import json
import unittest
from unittest.mock import patch, MagicMock

from selfcoder.config import Config
from selfcoder.llm import LLMClient, LLMError, parse_json


class JsonTests(unittest.TestCase):
    def test_wrapped_answer_and_reasoning(self):
        for text in (
            '<think>Example {"wrong": true}</think><answer>{"edits": []}</answer>',
            '```python\nexample\n```\n```json\n{"edits": []}\n```',
            'Example {invalid}; answer: {"edits": []} trailing {text}',
            '{"edits": [], "summary": "braces {inside} strings"}',
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_json(text)['edits'], [])

    def test_non_object_json_is_rejected(self):
        for text in ('[]', 'null', '"hello"'):
            with self.assertRaisesRegex(LLMError, 'not an object'):
                parse_json(text)

    def test_error_contains_response_preview(self):
        with self.assertRaisesRegex(LLMError, 'I cannot complete'):
            parse_json('I cannot complete this request')

    def chat(self, choice):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {'choices': [choice]}
        ).encode()
        with patch('urllib.request.urlopen', return_value=response):
            return LLMClient(Config(api_key_optional=True)).chat_json([])

    def test_truncation_is_reported(self):
        with self.assertRaisesRegex(LLMError, 'reached max_tokens'):
            self.chat({'finish_reason': 'length', 'message': {'content': '{'}})

    def test_reasoning_only_is_reported(self):
        with self.assertRaisesRegex(LLMError, 'reasoning_content but no final content'):
            self.chat({'message': {'content': '', 'reasoning_content': 'Thinking'}})

    def test_final_content_is_used(self):
        self.assertEqual(self.chat({'finish_reason': 'stop', 'message': {
            'content': '{"edits": []}', 'reasoning_content': 'Thinking',
        }}), {'edits': []})
