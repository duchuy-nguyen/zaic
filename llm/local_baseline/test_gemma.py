"""Verify Gemma's implicit BOS is included in the context guard."""
import time
import unittest
from .backend import LocalServer, BackendError


class GemmaTokenGuardTests(unittest.TestCase):
    def backend(self, model, context):
        server = LocalServer.__new__(LocalServer)
        server.model, server.context = model, context
        self.special = None
        self.generated = False
        def request(path, payload=None, **kwargs):
            if path == "/apply-template":
                return {"prompt": "test"}
            if path == "/tokenize":
                self.special = payload["add_special"]
                return {"tokens": [1] * (10 + int(self.special))}
            self.generated = True
            return {"usage": {"prompt_tokens": 10 + int(self.special)}}
        server.request = request
        return server

    def test_gemma_bos_can_exceed_context_at_boundary(self):
        server = self.backend("gemma", 20)
        with self.assertRaisesRegex(BackendError, "CONTEXT_OVERFLOW"):
            server.generate([], [], time.monotonic() + 30, max_tokens=10)
        self.assertTrue(self.special)
        self.assertFalse(self.generated)

    def test_existing_models_keep_original_tokenization(self):
        for model in ("qwen", "spark"):
            server = self.backend(model, 20)
            _, count = server.generate([], [], time.monotonic() + 30, max_tokens=10)
            self.assertFalse(self.special)
            self.assertTrue(self.generated)
            self.assertEqual(count, 10)


if __name__ == "__main__":
    unittest.main()
