import sys
sys.path.append('../research/qwen1_5')
from qwen1_5_tokenizer import Qwen2Tokenizer
from config.server_config import default_config, ServerConfig

if __name__ == '__main__':
    second_config = ServerConfig(default_config['model']['config']).config
    tokenizer = Qwen2Tokenizer(second_config['processor']['tokenizer']['vocab_file'],
                              second_config['processor']['tokenizer']['merges_file'],)
                              #eos_token="<|im_end|>", pad_token="<|im_end|>")
                              #bos_token="<|im_start|>", eos_token="<|im_end|>", pad_token="<|im_end|>")
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    print(f"BOS Token ID: {bos_token_id}, EOS Token ID: {eos_token_id}, Pad Token ID: {pad_token_id}")
    special_token = tokenizer.special_tokens
    print(special_token)

    prompt = "你是谁？"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="ms")
    token_ids = model_inputs['input_ids']
    print(f'Model_inputs: {token_ids}')
    print('=================')
    print(f'Default_chat_template: {tokenizer.default_chat_template}')      

    token_id = 200000
    token_content = tokenizer._convert_id_to_token(token_id)
    print(f'Token_ID: {token_id}, Token: {token_content}')