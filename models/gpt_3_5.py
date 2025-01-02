class GPT_3_5:
    api_key = "e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
    api_base = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer e3e9e07414a84058ba9d9bce313854634412c6cc36294e98ba3a382093b3959c"
        # Please change your KEY. If your key is XXX, the Authorization is "Authorization": "Bearer XXX"
    }
    # data = {
    #     "model": "gpt-3.5-turbo",
    #     # "gpt-3.5-turbo" version in gpt-3.5-turbo-1106,
    #     # "gpt-4" version in gpt-4-1106-version (gpt-4-vision-preview is NOT available in azure openai),
    #     # "gpt-3.5-turbo-16k", "gpt-4-32k"
    #     "messages": [{"role": "user", "content": "This is a test."}],
    #     "temperature": 0.7
    # }

