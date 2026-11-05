# Runtime tests offline

Từ repo root:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/ops -v
```

Dùng unittest, mock development và fake clock; không sleep, network, GPU hay data.
Kiểm dependency observation, order/isolation, không replay history, ask-stop,
schema/parse repair, empty-calls finish hợp lệ, single-call contract, deadline trước/sau backend và quanh validation/execute,
partial execution/unknown không retry, done=false, budgets và artifacts CLI.
Không xem mock là kiểm chứng ToolEnv hay submission chính thức.
