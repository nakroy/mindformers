import sys
sys.path.append('../research/qwen1_5')
from qwen1_5_tokenizer import Qwen2Tokenizer
from config.server_config import default_config, ServerConfig

def build_qwen_prompt(inputs: str):
    """Build qwen1.5 prompt template"""
    prompt = "{}"
    return prompt.format(inputs)

def build_multi_round_qwen(inputs, history):
    """Build multi round prompt for qwen1.5"""
    
    # Construct previous rounds for history
    multi_round_prompt = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>\n"
    prev_rounds = ""

    # Current round prompt
    current_prompt = f"<|im_start|>user\n{inputs}<|im_end|>\n<|im_start|>assistant\n"
    
    for i, (query, response) in enumerate(history):
        prev_rounds += multi_round_prompt.format(query, response)
    return prev_rounds + current_prompt

def split_inputs(inputs):
    split_token = '/system_prompt/'
    system_content = ''
    if split_token in inputs:
        system_content, inputs = inputs.split(split_token, 1)
    return system_content, inputs

def build_multi_round_qwen2(inputs, history):
    system_prompt_template = "<|im_start|>system\n{}<|im_end|>\n"
    system_content, inputs = split_inputs(inputs)
    system_prompt = ''
    if history == '':
        if system_content == '':
            system_prompt = system_prompt_template.format('You are a helpful assistant.')
        else:
            system_prompt = system_prompt_template.format(system_content)
    else:
        if system_content != '':
            system_prompt = system_prompt_template.format(system_content)

    current_prompt = system_prompt + f"<|im_start|>user\n{inputs}<|im_end|>\n<|im_start|>assistant\n"
    history = history + current_prompt
    return history

def create_chat_completion(messages):
    if messages[-1]["role"] != "user":
        raise ValueError
    query = messages[-1]["content"]

    prev_messages = messages[:-1]
    if len(prev_messages) > 0 and prev_messages[0]["role"] == 'system':
        query = prev_messages.pop(0)["content"] + query
    history = []
    if len(prev_messages) % 2 == 0:
        for i in range(0, len(prev_messages), 2):
            if prev_messages[i]["role"] == "user" and prev_messages[i+1]["role"] == "assistant":
                history.append([prev_messages[i]["content"], prev_messages[i+1]["content"]])
    return query, prev_messages, history

if __name__ == "__main__":
    second_config = ServerConfig(default_config['model']['config']).config
    tokenizer = Qwen2Tokenizer(second_config['processor']['tokenizer']['vocab_file'],
                              second_config['processor']['tokenizer']['merges_file'],)
    '''
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "你在说啥呢"
        }
    ]
    query, prev_messages, history = create_chat_completion(messages)
    print(f"query: {query}")
    print("====================")
    print(f"prev_messages: {prev_messages}")
    print("====================")
    print(f"history: {history}")

    '''
    history = ''
    max_length = 128
    while True:
        inputs = input()
        if inputs.strip() == "end":
            break
        prompted_inputs = build_qwen_prompt(inputs)
        inputs = build_multi_round_qwen2(prompted_inputs, history)
        outputs = f"{['k' for i in range(1024)]}"
        print(f'prompt inputs: {prompted_inputs}')
        print('=============')
        print(f'mulit_round_inputs: {inputs}')
        history = inputs + outputs + '<|im_end|>\n'
        print('==================')
        print(f'history: {history}')
        input_ids = tokenizer(inputs)["input_ids"]
        if len(input_ids) > max_length:
            input_ids = tokenizer('<|im_start|>assistant\n' + inputs.split('<|im_start|>assistant\n')[-1])["input_ids"]
        '''
        try:
            input_ids = tokenizer(inputs)["input_ids"]
            if len(input_ids) > max_length:
                raise ValueError(f"input_ids length {input_ids} exceeds the max length config {max_length}")
        except ValueError as e:
            print(f'Error: {e}')
            
        '''
        geneartion_kwargs = dict(
            input_ids = [input_ids]
        )

        print(f'Input ids: {input_ids}')
        print('==================')
        print(f'Generation kwargs: {geneartion_kwargs}')
        #print(f'input length is {len(geneartion_kwargs["input_ids"][0])}')
        print(f'input length is {len(input_ids)}')
