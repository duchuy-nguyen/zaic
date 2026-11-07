"""Prepared text/audio input boundary; never consumes transcript gold or labels."""
from copy import deepcopy
from typing import Protocol

from lifemate.runtime.contracts import Sample


class Transcriber(Protocol):
    def transcribe(self, audio: str, *, remaining_seconds: float) -> str: ...


def prepare_sample(sample: Sample, *, transcriber=None, remaining=lambda: 30.0) -> Sample:
    value = sample.model_input

    def text(turn, text_key, audio_key):
        if text_key in turn:
            result = turn[text_key]
        else:
            if transcriber is None:
                raise ValueError('ASR adapter required for audio input')
            if remaining() <= 0:
                raise TimeoutError('ASR deadline')
            audio = turn[audio_key]
            if not isinstance(audio, str) or not audio:
                raise ValueError('Invalid audio reference')
            result = transcriber.transcribe(audio, remaining_seconds=remaining())
        if not isinstance(result, str) or not result.strip():
            raise ValueError('Transcript must be nonempty text')
        return result

    context = value['context']
    clean = {k: deepcopy(context[k]) for k in (
        'current_time', 'timezone', 'location', 'current_location', 'user_profile') if k in context}
    installed = context['installed_apps']
    clean['installed_apps'] = {}
    for app, spec in installed.items():
        tools = spec['tools']
        if (not isinstance(app, str) or type(tools) is not list
                or any(not isinstance(t, str) or not t for t in tools)):
            raise ValueError('Invalid installed tools')
        clean['installed_apps'][app] = {'tools': list(tools)}
    history = []
    for turn in value.get('history', []):
        role = turn['role']
        if role == 'user':
            history.append({'role': 'user', 'text': text(turn, 'text', 'audio')})
        elif role in ('assistant', 'tool'):
            history.append({k: deepcopy(turn[k]) for k in (
                'role', 'text', 'tool_calls', 'tool_results') if k in turn})
        else:
            raise ValueError('Unsupported history role')
    return Sample(sample.sample_id, {'context': clean, 'history': history,
        'latest_text': text(value, 'latest_text', 'latest_audio')})


def permitted_names(sample: Sample, catalog: dict) -> set[str]:
    names = {t for app in sample.model_input['context']['installed_apps'].values()
             for t in app['tools']} | {'sys.ask_missing_info'}
    if names - catalog.keys():
        raise ValueError('Installed tool missing from catalog')
    return names
