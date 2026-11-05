"""Own a single loopback-only CUDA server; stop only the process created here."""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .setup_local import ROOT, TAG

MODELS = {
    "qwen": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "spark": "Spark-X2.5-4B-Q4_K_M.gguf",
    "gemma": "gemma-4-E2B-it-Q4_K_M.gguf",
    "qwen35": "Qwen3.5-4B-Q4_K_M.gguf",
    "qwen35_2b": "Qwen3.5-2B-Q4_K_M.gguf",
}


class BackendError(RuntimeError):
    pass


class LocalServer:
    def __init__(self, model: str, log_dir: Path, *, context: int = 16384):
        if not 1 <= context <= 16384:
            raise ValueError("Context must be within the competition limit")
        self.model = model
        self.context = context
        self.weights = ROOT / MODELS[model]
        record = json.loads((ROOT / ".runtime" / TAG / "installation.json").read_text(encoding="utf-8"))
        self.executable = Path(record["server"])
        self.log_dir = log_dir
        self.process = None
        self.log = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.command = []
        self.api_key = secrets.token_hex(24)

    def __enter__(self):
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.log_dir / "server.log").open("xb")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.command = [str(self.executable), "--model", str(self.weights), "--host", "127.0.0.1",
                        "--port", str(self.port), "--ctx-size", str(self.context), "--parallel", "1",
                        "--gpu-layers", "99", "--threads", "8", "--threads-batch", "8",
                        "--flash-attn", "on", "--jinja", "--reasoning", "off",
                        "--no-context-shift", "--no-webui", "--seed", "20261103", "--n-predict", "512",
                        "--api-key", self.api_key]
        try:
            self.process = subprocess.Popen(self.command, cwd=self.executable.parent, stdout=self.log,
                                            stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            until = time.monotonic() + 180
            while time.monotonic() < until:
                if self.process.poll() is not None:
                    raise BackendError(f"Server exited with {self.process.returncode}; inspect {self.log_dir / 'server.log'}")
                try:
                    self.request("/health", timeout=2)
                    return self
                except (BackendError, TimeoutError, OSError):
                    time.sleep(0.4)
            raise BackendError("Server startup exceeded 180 seconds")
        except BaseException:
            self.close()
            raise

    def request(self, path: str, payload: dict | None = None, *, timeout: float = 30):
        if timeout <= 0:
            raise TimeoutError("Sample deadline exhausted")
        request = urllib.request.Request(self.url + path,
                                         data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise BackendError(f"HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("Local model request timed out") from error
            raise BackendError(str(error)) from error

    def generate(self, messages: list, tools: list, deadline: float, *, max_tokens: int = 512) -> tuple[dict, int]:
        payload = {"messages": messages, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": False,
                   "temperature": 0, "top_k": 1, "seed": 20261103, "max_tokens": max_tokens,
                   "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        rendered = self.request("/apply-template", payload, timeout=deadline - time.monotonic())
        tokenized = self.request("/tokenize", {"content": rendered["prompt"], "add_special": self.model == "gemma", "parse_special": True},
                                 timeout=deadline - time.monotonic())
        prompt_tokens = len(tokenized["tokens"])
        if prompt_tokens + max_tokens > self.context:
            raise BackendError(f"CONTEXT_OVERFLOW: {prompt_tokens} input + {max_tokens} reserve > {self.context}; no truncation")
        response = self.request("/v1/chat/completions", payload, timeout=deadline - time.monotonic())
        return response, prompt_tokens

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()

    def public_command(self):
        return ["<ephemeral-local-key>" if index and self.command[index - 1] == "--api-key" else item
                for index, item in enumerate(self.command)]

    def __exit__(self, *args):
        self.close()
