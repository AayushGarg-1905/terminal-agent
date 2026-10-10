import tiktoken

## Refer from openai cookbook
def get_tokenizer(model:str):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return encoding.encode
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return encoding.encode

def count_tokens(text:str,model:str = "gpt-4"):
    tokenizer = get_tokenizer(model)
    if tokenizer:
        return len(tokenizer(text))
    else:
        return max(1,len(text)//4)