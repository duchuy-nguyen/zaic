"""Optional maintainer check against pinned public tokenizer/template files only.

No model weights or BTC data. Requires standalone tokenizers + jinja2.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

from core import assistant, training_tokens, parse_action, write_json


def main():
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2 import TemplateError
    root = Path(__file__).resolve().parents[3] / 'runs/oneclick_template_checks'
    report = []
    catalog = {'fixture.write': {'parameters': {'properties': {'value': {'type': 'integer'},
        'enabled': {'type': 'boolean'}, 'items': {'type': 'array'}, 'text': {'type': 'string'}}}}}
    tools = [{'type': 'function', 'function': {'name': name, 'parameters': spec['parameters']}} for name, spec in catalog.items()]
    call = {'name': 'fixture.write', 'arguments': {'value': 1, 'enabled': False, 'items': [1, 2], 'text': 'đèn phòng khách'}}
    for role, model_id, revision in [
        ('student', 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
        ('teacher', 'Qwen/Qwen3.5-9B', 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'),
        ('legacy_teacher', 'unsloth/Qwen3.8-27B-unsloth-bnb-4bit', '8aa5f05d26b7205477066e1449e0af13f762a299')]:
        directory = root / (role + '_' + revision[:12]); directory.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json'):
            path = directory / name
            if not path.exists():
                with urllib.request.urlopen(f'https://huggingface.co/{model_id}/resolve/{revision}/{name}', timeout=60) as response:
                    path.write_bytes(response.read())
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                           extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        def raise_exception(message): raise TemplateError(message)
        env.globals['raise_exception'] = raise_exception
        template = env.from_string((directory / 'chat_template.jinja').read_text(encoding='utf-8'))
        tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        model_config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        checkpoint_eos = model_config.get('text_config', {}).get('eos_token_id', model_config.get('eos_token_id'))
        chat_eos = tokenizer.token_to_id('<|im_end|>')
        class Wrapper:
            def apply_chat_template(self, messages, tokenize, **kwargs):
                return template.render(messages=messages, **kwargs)
            def encode(self, text, **kwargs): return tokenizer.encode(text, add_special_tokens=False).ids
        messages = [{'role': 'system', 'content': 'Fixture only.'}, {'role': 'user', 'content': 'fixture'}]
        cases = []
        for target in (assistant(call), assistant(None)):
            record = {'messages': messages, 'tools': tools, 'target': target}
            encoded, length = training_tokens(Wrapper(), record, 16384)
            target_ids = [value for value in encoded['labels'] if value != -100]
            text = tokenizer.decode(target_ids, skip_special_tokens=False).split('<|im_end|>')[0].strip()
            parsed = parse_action(text, catalog)
            assert parsed == (call if target.get('tool_calls') else None)
            cases.append({'tokens': length, 'target_tokens': len(target_ids), 'roundtrip_ok': True})
        report.append({'role': role, 'model': model_id, 'revision': revision, 'files': hashes,
                       'checkpoint_eos': checkpoint_eos, 'chat_eos': chat_eos, 'cases': cases})
    write_json(root / 'report.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__': main()
