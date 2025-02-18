# """
# Written by <Geng He>
# """
#
import torch

print("Test 1: Torch available or not.")
print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# def LLaMa():
#     # Load model directly
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#
#     tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Python-hf")
#     model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-13b-Python-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
#
#     model.to("cuda:0")
#
# if __name__ == '__main__':
#     print(LLaMa())

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# import torch
# from PIL import Image
# import requests
#
# processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")
#
# model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
# model.to("cuda:0")
#
# # prepare image and text prompt, using the appropriate prompt template
# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
#
# # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image")
# conversation = [
#     {
#
#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What is shown in this image?"},
#           {"type": "image"},
#         ],
#     },
# ]
# print('--------------------------------------------------------------------')
# print(processor.chat_template)
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
#
# inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
#
# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)
#
# print(processor.decode(output[0], skip_special_tokens=True))