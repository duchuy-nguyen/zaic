"""Build ONE portable Kaggle notebook embedding all project-owned workers.

No models/datasets/official runtime are embedded. Rebuild after editing workers.
"""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

HERE = Path(__file__).resolve().parent


def md(value):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(value).strip().splitlines(keepends=True)}


def code(value):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': textwrap.dedent(value).strip().splitlines(keepends=True)}


def build():
    sources = {name: (HERE / 'oneclick' / name).read_text(encoding='utf-8') for name in ('core.py', 'worker.py', 'bounded_attention.py', 'verify_training_stack.py')}
    sources['scoring.py'] = (HERE.parent / 'local_baseline/scoring.py').read_text(encoding='utf-8')
    raw = json.dumps(sources, ensure_ascii=False, sort_keys=True).encode()
    payload = base64.b64encode(zlib.compress(raw, 9)).decode()
    bundle_sha = hashlib.sha256(raw).hexdigest()
    cells = [md('''
        # LifeMate — One Notebook: ASR LoRA + Teacher Distillation + Student QLoRA

        **Teacher:** Qwen3.5-9B, NF4 at load. Teacher 27B bị OOM/CPU spill trên hai T4 của team.
        **Student:** Qwen3.5-4B. **ASR:** PhoWhisper-medium.

        Notebook tự chứa code, không cần clone repo hay upload Python scripts.
        Chỉ attach official data kit BTC trong phạm vi truy cập được phép. Notebook/data/outputs
        phải Private. Không publish traces, gold hoặc dataset BTC lên Hub/public notebook.

        Mặc định `MODE='pilot'`: chạy THẬT đủ các stages trên subset và ít optimizer steps,
        không phải benchmark đầy đủ. Sau khi pilot pass, đổi MODE='full', dùng RUN_NAME mới.
        Một file có thể cần nhiều session GPU; checkpoint/resume không bảo đảm hoàn tất trong 12h.

        Chưa được chứng nhận chạy trên Kaggle GPU. Notebook sẽ dừng nếu GPU, memory,
        coverage, teacher quality hoặc pinned stack không đạt gate; không tự đổi model,
        truncate, đọc hidden state hay bỏ calls để chạy tiếp.
        '''), code('''
        # Chỉnh cell này. Không dán token vào notebook.
        MODE = 'pilot'                  # 'pilot' hoặc 'full'
        RUN_NAME = 'lifemate_pilot_v6'   # đổi khi đổi MODE/model/config/code
        DATA_ROOT = ''                  # auto-detect; hoặc folder CHỨA train/
        INPUT_ARCHIVE = ''              # auto-detect train+public_test.zip nếu chưa giải nén
        RESUME_FROM = ''                # optional: previous private run folder hoặc *_resume.tar.gz input
        REUSE_ASR_FROM = ''             # optional: run cũ có ASR stages hoàn tất; verify rồi reuse, không train lại
        REUSE_TEACHER_FROM = ''         # optional: teacher/examples đã hoàn tất, verify rồi reuse
        HF_SECRET_NAME = ''             # optional Kaggle Secret name; public models không bắt buộc token
        TEACHER_ID = 'Qwen/Qwen3.5-9B'
        TEACHER_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        TEACHER_PREQUANTIZED = False
        TEACHER_MIN_GPUS = 2
        TEACHER_GPU_MEMORY_FRACTION = 0.85  # budget weights; giữ chỗ cho activations/cache
        TEACHER_PREFILL_CHUNK_SIZE = 256   # xử lý đủ prompt theo chunks, không truncate
        STUDENT_CONTEXT = 8192           # coverage/real-shape smoke gate, không truncate
        ATTENTION_QUERY_CHUNK = 128      # exact attention, bounded score tiles + backward recomputation
        STUDENT_LOSS_CHUNK = 32          # bounded target x vocab logits, exact masked CE
        STUDENT_AMP_INIT_SCALE = 128.0   # same GradScaler policy in precheck/smoke/Trainer
        STUDENT_COMPACT_FROZEN_IO = True # frozen embedding/head FP16; adapters/norms retain FP32
        MIN_TRAINING_COVERAGE = 0.80     # xem length_report.json; không hạ chỉ để bỏ qua lỗi
        ALLOW_EXISTING_CACHE_SPACE_OVERRIDE = False  # chỉ True nếu đã kiểm cache/disk thật
        SESSION_BUDGET_HOURS = 8.5       # chừa thời gian archive/output nếu session chỉ có 10h
        '''), md('''
        ## Bootstrap và dependencies

        Kaggle: GPU T4 x2, Internet ON, notebook Private. Model weights/cache dùng `/tmp`,
        adapters/checkpoints/reports lưu `/kaggle/working/lifemate/<RUN_NAME>`.
        Venv cô lập, cài CUDA PyTorch 2.9.1 từ index chính thức; không sửa packages của kernel.
        Worker mới cho mỗi stage giúp giải phóng teacher trước khi train ASR/student tiếp.
        Hỗ trợ Python 3.11–3.13; kiểm tra CUDA allocation trước khi tải model weights.
        '''), code('''
        import os, sys, subprocess, json, pathlib, shutil, hashlib, base64, zlib, time
        from pathlib import Path
        if MODE not in ('pilot', 'full'):
            raise ValueError('MODE must be pilot/full')
        if os.name != 'posix' or not Path('/kaggle/working').is_dir():
            raise RuntimeError('Notebook training targets Kaggle Linux; do not Run All on local Windows')
        if not RUN_NAME or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in RUN_NAME):
            raise ValueError('Unsafe RUN_NAME')
        os.environ.update(HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                          CUBLAS_WORKSPACE_CONFIG=':4096:8', WANDB_DISABLED='true')
        SESSION_DEADLINE = time.time() + SESSION_BUDGET_HOURS * 3600
        HF_HOME = Path('/tmp/lifemate_hf'); HF_HOME.mkdir(parents=True, exist_ok=True)
        os.environ['HF_HOME'] = str(HF_HOME)
        if HF_SECRET_NAME:
            from kaggle_secrets import UserSecretsClient
            os.environ['HF_TOKEN'] = UserSecretsClient().get_secret(HF_SECRET_NAME)
        if not (3, 11) <= sys.version_info[:2] <= (3, 13):
            raise RuntimeError('Pinned training stack supports Python 3.11–3.13; current: ' + sys.version)
        # Kaggle may inject user packages/sitecustomize via PYTHONPATH. Do not inherit
        # those hooks into training workers or depend on kernel-local torch visibility.
        TRAIN_ENV = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        TRAIN_ENV['PYTHONNOUSERSITE'] = '1'
        smi = shutil.which('nvidia-smi')
        if not smi:
            raise RuntimeError('No NVIDIA driver found; enable Kaggle GPU T4 x2 first')
        subprocess.run([smi, '--query-gpu=name,driver_version', '--format=csv,noheader'],
                       check=True, env=TRAIN_ENV)
        VENV = Path('/tmp/lifemate_training_env_v2')
        # Reapply without clearing existing packages: an earlier ensurepip failure
        # can leave bin/python present while pip is absent. Never invoke ensurepip.
        subprocess.run([sys.executable, '-I', '-m', 'venv', '--without-pip', str(VENV)],
                       check=True, env=TRAIN_ENV)
        venv_config = (VENV / 'pyvenv.cfg').read_text(encoding='utf-8').lower()
        if 'include-system-site-packages = false' not in venv_config:
            raise RuntimeError('Expected isolated v2 venv; preserve artifacts and use a fresh session')
        PYTHON = str(VENV / 'bin/python')
        pip_probe = subprocess.run([PYTHON, '-I', '-m', 'pip', '--version'],
                                   env=TRAIN_ENV, capture_output=True, text=True)
        if pip_probe.returncode:
            from urllib.request import urlopen
            bootstrap = VENV / 'get-pip.py'
            bootstrap_url = 'https://bootstrap.pypa.io/get-pip.py'
            with urlopen(bootstrap_url, timeout=120) as response:
                bootstrap_bytes = response.read()
            compile(bootstrap_bytes, str(bootstrap), 'exec')
            bootstrap.write_bytes(bootstrap_bytes)
            print('Bootstrapping pip without ensurepip:', bootstrap_url,
                  'sha256=', hashlib.sha256(bootstrap_bytes).hexdigest(), flush=True)
            subprocess.run([PYTHON, '-I', str(bootstrap), 'pip==25.3', '--only-binary=:all:',
                            '--no-cache-dir', '--index-url', 'https://pypi.org/simple'],
                           check=True, env=TRAIN_ENV)
        subprocess.run([PYTHON, '-I', '-m', 'pip', '--version'], check=True, env=TRAIN_ENV)
        TORCH_PINS = ['torch==2.9.1', 'torchvision==0.24.1', 'torchaudio==2.9.1']
        constraints = VENV / 'torch-constraints.txt'
        constraints.write_text('torch==2.9.1+cu126\\ntorchvision==0.24.1+cu126\\ntorchaudio==2.9.1+cu126\\n', encoding='utf-8')
        PINS = ['transformers==5.20.2', 'peft==0.21.2', 'accelerate==1.15.0',
                'bitsandbytes==0.50.2', 'huggingface-hub==2.3.0', 'tokenizers==0.23.3',
                'safetensors==0.8.0', 'numpy==2.2.6', 'librosa==0.11.0', 'soundfile==0.13.1',
                'jinja2==3.1.6', 'scikit-learn==1.6.1', 'scipy==1.15.3', 'pandas==2.2.3', 'wrapt==1.17.3']
        subprocess.run([PYTHON, '-I', '-m', 'pip', 'install', '--only-binary=:all:', '--no-cache-dir',
            '--index-url', 'https://download.pytorch.org/whl/cu126', *TORCH_PINS], check=True, env=TRAIN_ENV)
        subprocess.run([PYTHON, '-I', '-m', 'pip', 'install', '--only-binary=:all:', '--no-cache-dir',
            '--upgrade', '--upgrade-strategy', 'only-if-needed', '-c', str(constraints), *PINS], check=True, env=TRAIN_ENV)
        subprocess.run([PYTHON, '-I', '-c',
            'import sys, torch, torchvision, torchaudio, transformers, peft, bitsandbytes; '
            'assert sys.prefix != sys.base_prefix, "Expected isolated venv"; '
            'assert torch.__version__ == "2.9.1+cu126", torch.__version__; '
            'assert torch.version.cuda == "12.6", "Expected CUDA build, not CPU torch"; '
            'assert torch.cuda.is_available(), "CUDA unavailable: check GPU selection/NVIDIA driver"; '
            'x = torch.ones(4, device="cuda"); assert x.sum().item() == 4; '
            'print({"python": sys.version, "torch": torch.__version__, "torch_file": torch.__file__, '
            '"cuda": torch.version.cuda, "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})'],
            check=True, env=TRAIN_ENV)
        '''), code(f'''
        # Embedded project source. No official dataset/runtime or credentials in payload.
        _PAYLOAD = {payload!r}
        _RAW = zlib.decompress(base64.b64decode(_PAYLOAD))
        BUNDLE_SHA256 = {bundle_sha!r}
        if hashlib.sha256(_RAW).hexdigest() != BUNDLE_SHA256:
            raise ValueError('Notebook bundle checksum mismatch')
        SOURCE = Path('/tmp/lifemate_notebook_src'); SOURCE.mkdir(parents=True, exist_ok=True)
        for name, content in json.loads(_RAW).items():
            if name not in ('core.py', 'worker.py', 'scoring.py', 'bounded_attention.py', 'verify_training_stack.py'):
                raise ValueError('Unexpected embedded file')
            compile(content, name, 'exec')
            (SOURCE / name).write_text(content, encoding='utf-8')
        sys.path.insert(0, str(SOURCE))
        from core import extract_archive, restore_archive, digest
        print('Verified embedded workers:', BUNDLE_SHA256)
        '''), code('''
        # Find kit without inspecting initial app state. ToolEnv owns app-state loading.
        input_root = Path('/kaggle/input')
        root = Path(DATA_ROOT).resolve() if DATA_ROOT else None
        if root is None:
            manifests = [p for p in input_root.rglob('train.jsonl') if p.parent.name == 'train' and '__MACOSX' not in p.parts]
            roots = sorted({p.parent.parent for p in manifests})
            if len(roots) == 1:
                root = roots[0]
            elif len(roots) > 1:
                raise ValueError('Multiple train kits: set DATA_ROOT explicitly')
        if root is None:
            archives = [Path(INPUT_ARCHIVE)] if INPUT_ARCHIVE else list(input_root.rglob('train+public_test.zip'))
            if len(archives) != 1:
                raise ValueError('Attach exactly one kit ZIP or set DATA_ROOT/INPUT_ARCHIVE')
            archive = archives[0]
            archive_sha = digest(archive)
            extracted = Path('/tmp/lifemate_official_kit')
            stamp = extracted / 'archive.sha256'
            if not stamp.is_file() or stamp.read_text().strip() != archive_sha:
                if extracted.exists() and any(extracted.iterdir()):
                    raise ValueError('Extraction cache differs; use a fresh session/cache rather than overwrite')
                extracted.mkdir(parents=True, exist_ok=True)
                extract_archive(archive, extracted)
                stamp.write_text(archive_sha, encoding='utf-8')
            manifests = [p for p in extracted.rglob('train.jsonl') if p.parent.name == 'train']
            if len(manifests) != 1:
                raise ValueError('Unrecognized official kit layout')
            root = manifests[0].parent.parent
        for name in ('train/train.jsonl', 'train/train_tools_catalog.json', 'train/zaic_tools/__init__.py'):
            if not (root / name).is_file():
                raise FileNotFoundError('Missing official kit component: ' + name)
        for name in ('train/audio', 'train/train_environment'):
            if not (root / name).is_dir():
                raise FileNotFoundError('Missing official kit directory: ' + name)
        RUN = Path('/kaggle/working/lifemate') / RUN_NAME
        if RESUME_FROM:
            previous = Path(RESUME_FROM).resolve()
            if RUN.exists():
                raise ValueError('Resume destination exists; keep RESUME_FROM empty to continue existing working run')
            if previous.is_file():
                previous = restore_archive(previous, Path('/tmp/lifemate_resume') / digest(previous)[:16], RUN_NAME)
            if not (previous / 'config.json').is_file():
                raise ValueError('RESUME_FROM must be the previous run folder containing config.json')
            shutil.copytree(previous, RUN)
        RUN.mkdir(parents=True, exist_ok=True)
        limits = {'fit_samples': 64, 'val_samples': 8, 'teacher_samples': 12,
                  'asr_max_steps': 20, 'student_max_steps': 20} if MODE == 'pilot' else {
                  'fit_samples': None, 'val_samples': None, 'teacher_samples': None,
                  'asr_max_steps': -1, 'student_max_steps': -1}
        cfg = dict(mode=MODE, run_dir=str(RUN), data_root=str(root), bundle_sha256=BUNDLE_SHA256,
            reuse_asr_from=REUSE_ASR_FROM,
            reuse_teacher_from=REUSE_TEACHER_FROM,
            bundle_files=sorted(json.loads(_RAW)),
            attention_query_chunk=ATTENTION_QUERY_CHUNK, student_loss_chunk=STUDENT_LOSS_CHUNK,
            student_amp_init_scale=STUDENT_AMP_INIT_SCALE,
            student_compact_frozen_io=STUDENT_COMPACT_FROZEN_IO,
            bootstrap_version='isolated-cu126-no-ensurepip-v3',
            seed=20261105, deterministic=True, val_fraction=.20, limits=limits,
            teacher_id=TEACHER_ID, teacher_revision=TEACHER_REVISION,
            teacher_base_id=TEACHER_ID, teacher_prequantized=TEACHER_PREQUANTIZED,
            teacher_min_gpus=TEACHER_MIN_GPUS, teacher_context=16384, teacher_seconds=180,
            teacher_gpu_memory_fraction=TEACHER_GPU_MEMORY_FRACTION,
            teacher_prefill_chunk_size=TEACHER_PREFILL_CHUNK_SIZE,
            teacher_min_score=100., minimum_teacher_accepted=1 if MODE == 'pilot' else 10,
            student_id='Qwen/Qwen3.5-4B', student_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
            student_train_context=STUDENT_CONTEXT, inference_context=16384, max_new_tokens=512,
            minimum_training_coverage=MIN_TRAINING_COVERAGE, student_lora_r=8,
            student_grad_accum=4 if MODE == 'pilot' else 8, student_lr=1e-4, student_epochs=2,
            asr_id='vinai/PhoWhisper-medium', asr_revision='55a7e3eb6c906de891f8f06a107754427dd3be79',
            asr_lora_r=16, asr_grad_accum=4 if MODE == 'pilot' else 8, asr_lr=1e-4,
            asr_epochs=2, asr_max_new_tokens=256, asr_eval_seconds=60,
            save_steps=5 if MODE == 'pilot' else 50, max_steps=18, max_calls=10, agent_seconds=30,
            strict_reference_consistency=True,
            expected_train_sha256='24c42c79fbe8db932fe9817d041c0d0ca422ac8177b413cc1be0cda1fd8941a8',
            allow_existing_cache_space_override=ALLOW_EXISTING_CACHE_SPACE_OVERRIDE)
        config_path = RUN / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Run configuration/code/input paths changed; choose a new RUN_NAME')
        config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        (RUN / 'source').mkdir(exist_ok=True)
        for name in ('core.py', 'worker.py', 'scoring.py', 'bounded_attention.py', 'verify_training_stack.py'):
            shutil.copy2(SOURCE / name, RUN / 'source' / name)
        def save_resume_archive():
            target = shutil.make_archive(str(RUN.parent / (RUN_NAME + '_resume')), 'gztar',
                root_dir=RUN.parent, base_dir=RUN.name)
            print('PRIVATE resume archive:', target, flush=True)
            return target
        SESSION_PAUSED = False
        def run_stage(stage):
            global SESSION_PAUSED
            if SESSION_PAUSED:
                print('PAUSED: skip remaining stage', stage, flush=True)
                return
            print('STAGE:', stage, flush=True)
            from collections import deque
            error_tail = deque(maxlen=100)
            env = dict(TRAIN_ENV, PYTHONHASHSEED=str(cfg['seed']), PYTHONPATH=str(SOURCE),
                       LIFEMATE_SESSION_DEADLINE=str(SESSION_DEADLINE))
            if stage in ('stack_check', 'student_precheck', 'asr_base', 'asr_train', 'asr_tuned', 'student_base', 'student_smoke',
                         'student_train', 'student_text', 'student_audio'):
                # Trainer otherwise detects both GPUs and may invoke DataParallel.
                env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
            logs = RUN / 'logs'; logs.mkdir(exist_ok=True)
            with (logs / (stage + '.log')).open('a', encoding='utf-8') as log:
                process = subprocess.Popen([PYTHON, '-I', '-u', str(SOURCE / 'worker.py'), stage,
                    '--config', str(config_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace', env=env)
                try:
                    for line in process.stdout:
                        error_tail.append(line)
                        print(line, end='', flush=True); log.write(line); log.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    save_resume_archive()
                    raise
            if code:
                if code == 75:
                    save_resume_archive()
                    SESSION_PAUSED = True
                    print('SESSION PAUSED SAFELY: checkpoints/archive saved. Remaining stages skipped so Save Version can finish normally. Resume unchanged config next session.', flush=True)
                    return
                try:
                    save_resume_archive()
                except Exception as archive_error:
                    print('Could not archive failed run:', type(archive_error).__name__, str(archive_error), flush=True)
                marker = RUN / ('stage_' + stage + '.json')
                reason = ''
                if marker.is_file():
                    try:
                        reason = json.loads(marker.read_text(encoding='utf-8')).get('error', '')
                    except (ValueError, OSError):
                        pass
                raise RuntimeError('Stage failed: ' + stage + ' (exit ' + str(code) + '). '
                    + reason + '\\nLog: ' + str(logs / (stage + '.log'))
                    + '\\n--- Worker log tail ---\\n' + ''.join(error_tail)[-12000:])
        print('Mode:', MODE, '| output:', RUN, '| official TRAIN only')
        '''), md('''
        ## 1. Kiểm tra môi trường, split và gold references

        Chỉ train BTC được dùng. Group shared audio bytes, exact/near utterances trước khi
        tách; hash/manifests được freeze. Không đảm bảo speaker-disjoint vì thiếu speaker IDs.
        Reference replay là job train-only riêng; model không thấy gold/current labels/state.
        '''), code("run_stage('preflight')\nrun_stage('stack_check')\nrun_stage('prepare')\nrun_stage('reuse_asr')\nrun_stage('references')\nrun_stage('reuse_teacher')\nrun_stage('student_precheck')"),
        md('''
        ## 2. PhoWhisper-medium: baseline → LoRA → validation

        Audio latest và user history đi cùng gold transcript thật. Không dùng synthetic
        transcript đổi chữ nhưng giữ WAV gốc. WER/CER giữ case/punctuation và tính lỗi là deletions.
        Giữ ASR nguyên bản nếu LoRA không cải thiện WER/CER trên cùng validation.
        '''), code("run_stage('asr_base')\nrun_stage('asr_train')\nrun_stage('asr_tuned')"),
        md('''
        ## 3. Teacher Qwen3.5-9B: rollout train-fit và lọc

        Teacher được quantize NF4 lúc load; inference shard hai GPU, không có CPU/disk offload tự động.
        Non-thinking, một tool/bước, actual observations, exactly one ToolEnv/sample/run.
        Schema/state/local-call-F1 gate mặc định 100; scorer là local approximation, không official.
        Không dùng public/private để sinh nhãn. Teacher và student không cùng load.
        '''), code("run_stage('teacher')\nrun_stage('examples')"),
        md('''
        ## 4. Student: baseline → real-shape memory smoke → QLoRA → text/audio eval

        Train một lượt bằng gold + teacher data đã lọc. Exact duplicate targets được dedup;
        xem examples_report để biết teacher có thực sự bổ sung examples mới hay chỉ trùng gold.
        Đây không thay cho ablation gold-only riêng nếu muốn chứng minh lợi ích distillation.

        Không cắt target/schema để vừa context. Các examples quá dài được liệt kê và coverage
        phải vượt gate. Mask prompt/tool observations; compute lm_head chỉ ở vị trí dự đoán target
        để tránh logits tensor [8K × 248K] đầy đủ. QLoRA single GPU; GPU thứ hai không tự cộng VRAM.
        Audio eval dùng ASR được chọn và student cùng process, không fallback gold transcript.
        '''), code("run_stage('student_smoke')\nrun_stage('student_base')\nrun_stage('student_train')\nrun_stage('student_text')\nrun_stage('student_audio')"),
        md('''
        ## 5. Export và xem kết quả

        Output là ASR/student adapters, tokenizer/processor, fixed split, configs và metrics.
        Base model weights vẫn cần được đóng gói riêng cho offline inference. Không có teacher
        trong bộ inference cuối. Chưa tạo Docker/submission public/private từ notebook training này.
        '''), code('''
        run_stage('export')
        if SESSION_PAUSED:
            print('INCOMPLETE RUN: session budget reached; this is a resumable pause, not completed training.')
        else:
            report = json.loads((RUN / 'FINAL_REPORT.json').read_text(encoding='utf-8'))
            print(json.dumps(report, ensure_ascii=False, indent=2))
        print('PRIVATE outputs:', RUN)
        save_resume_archive()
        '''), md('''
        ## Resume / giới hạn

        Cùng session: Run All lại với config/RUN_NAME không đổi sẽ verify và skip stages đã xong.
        Session mới: lưu outputs ở chế độ Private, attach lại, đặt RESUME_FROM tới run folder
        hoặc file `*_resume.tar.gz`; notebook tự giải nén archive của mình an toàn.
        Checkpoints optimizer resume được nếu nguồn/model/config/code vẫn giống nhau.
        Session budget mặc định 8.5h dừng ở sample/optimizer boundary, tự tạo resume archive,
        skip stages còn lại và kết thúc notebook bình thường để Save Version lưu outputs.
        Chọn Save & Run All và bật Save Output nếu UI có tùy chọn; kiểm tra Output sau khi run xong.
        Chạy interactive thì cần tự lưu private archive/folder ra storage bền vững.

        Nếu có `.inflight.json` chưa có committed record, có thể ToolEnv đã thay đổi state.
        Notebook dừng để tránh reset/rerun âm thầm; giữ artifacts và review sample đó.
        Pilot không chứng minh runtime BTC hoặc full-run duration. Context dài/T4 có thể OOM;
        notebook giữ log và dừng. Không tự đổi teacher, dtype hoặc hạ coverage.

        Nguồn: [Teacher Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B),
        [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B),
        [PhoWhisper](https://huggingface.co/vinai/PhoWhisper-medium),
        [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
        [PEFT](https://huggingface.co/docs/peft/developer_guides/quantization).
        ''')]
    for i, cell in enumerate(cells): cell['id'] = f'lifemate-{i:02d}'
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'lifemate': {'bundle_sha256': bundle_sha, 'default_mode': 'pilot', 'gpu_tested': False}},
        'nbformat': 4, 'nbformat_minor': 5}
    target = HERE / 'LifeMate_OneClick.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
