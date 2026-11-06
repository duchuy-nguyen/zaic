"""Optional CPU compatibility check using random tiny weights, not BTC data.

Run with Transformers 5.20.2 and PyTorch installed. Does not download model weights.
GPU memory fitness of the real 27B teacher must still be checked on Kaggle.
"""
def main():
    import torch
    import transformers
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    if transformers.__version__ != '5.20.2':
        raise RuntimeError('Use the pinned Transformers 5.20.2 stack')
    torch.manual_seed(1)
    torch.set_num_threads(1)
    config = Qwen3_5Config(text_config=dict(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=['linear_attention', 'full_attention'], max_position_embeddings=256,
        bos_token_id=1, eos_token_id=127, pad_token_id=0), vision_config=dict(
        depth=1, hidden_size=32, intermediate_size=64, num_heads=4, patch_size=2,
        spatial_merge_size=1, temporal_patch_size=1, out_hidden_size=32,
        deepstack_visual_indexes=[]))
    model = Qwen3_5ForConditionalGeneration(config).cpu().eval()
    for length in (17, 33):
        ids = torch.randint(2, 63, (1, length))
        options = dict(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3,
            min_new_tokens=3, do_sample=False, use_cache=True, pad_token_id=0,
            eos_token_id=127, logits_to_keep=1)
        with torch.inference_mode():
            full = model.generate(**options)
            chunked = model.generate(**options, prefill_chunk_size=16)
        if not torch.equal(full, chunked):
            raise AssertionError('Greedy outputs differ at input length ' + str(length))
    print('PASS: random tiny hybrid Qwen3.5 CPU; chunked/full prefill outputs equal at lengths 17/33')


if __name__ == '__main__':
    main()
