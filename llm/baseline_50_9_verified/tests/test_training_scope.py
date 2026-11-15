"""Public source must not quietly include a different model's training branch."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TrainingScopeTests(unittest.TestCase):
    def test_no_opposite_model_in_training_source(self):
        forbidden=("Qwen3ForCausalLM","Qwen/Qwen3-4B","PhoWhisper-medium",
                   "gipformer_val","from local_baseline import","from common import")
        for path in (ROOT/"training_reference").glob("*.py"):
            source=path.read_text(encoding="utf8")
            for value in forbidden:
                self.assertNotIn(value,source,path.name)
            tree=ast.parse(source)
            for node in tree.body:
                if isinstance(node,ast.If) and path.name in {"qwen35_training.py","phowhisper_large_training.py"}:
                    self.assertNotIn("__main__",ast.unparse(node.test),path.name)
    def test_reference_modules_require_explicit_private_binding(self):
        q=importlib.import_module("training_reference.qwen35_training")
        a=importlib.import_module("training_reference.phowhisper_large_training")
        self.assertIsNone(q.HERE)
        self.assertIsNone(a.HERE)
        with self.assertRaisesRegex(RuntimeError,"bind_private_workspace"):
            q.training(None,None,False,False)
        with self.assertRaisesRegex(RuntimeError,"bind_private_workspace"):
            a.training(False,False)
    def test_completion_helpers_keep_exact_source_function_ast(self):
        path=ROOT/"training_reference/completion_loss.py"
        tree=ast.parse(path.read_text(encoding="utf8"))
        names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(names,{"collate_rows","completion_causal_loss"})
        expected=json.loads((ROOT/"training_reference/DERIVATION.json").read_text())["exact_helper_function_ast_sha256"]
        for qualified,digest in expected.items():
            filename,function=qualified.split(":")
            source=ast.parse((ROOT/"training_reference"/filename).read_text(encoding="utf8"))
            node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==function)
            actual=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            self.assertEqual(actual,digest,qualified)
