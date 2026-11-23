"""Actual native chat-template encoding and completion-only loss mask."""

def encode(example, tokenizer):
    kwargs = {'tools': example['tools'], 'tokenize': True, 'return_dict': False, 'enable_thinking': False}
    prompt = tokenizer.apply_chat_template(example['prompt'], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(example['prompt'] + example['completion'], add_generation_prompt=False, **kwargs)
    if full[:len(prompt)] != prompt:
        raise ValueError('Prompt is not a token prefix; refuse incorrect loss mask: ' + example['id'])
    if not 0 < len(prompt) < len(full) <= 16384:
        raise ValueError(f"Empty target/context overflow {example['id']}: prompt={len(prompt)}, total={len(full)}; no truncation")
    return {'id': example['id'], 'source_id': example['source_id'], 'input_ids': full, 'labels': [-100] * len(prompt) + full[len(prompt):], 'length': len(full), 'target_tokens': len(full) - len(prompt), 'kind': example['kind']}
