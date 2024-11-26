
import torch
from huggingface_hub import login
login(token='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')
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    device='cuda'
)


prompt = """
    You are an expert in Python code annotation.
Next, I will provide you with some code comments in different styles, some of which may be extremely colloquial. I hope you can help me turn them into code comments with a unified style of 'Requirements'.
Example 1:
User input: # How to drop rows of Pandas DataFrame whose value in a certain column is NaN
Your output:
# Drop rows of Pandas DataFrame whose value in a certain column is NaN
Example 2:
User input:
# list_of_lists format: [header, [row1], [row2], ...]
# header format: [column1, column2, ...]
# row format: [value1, value2, ...]
# How to convert list to dataframe?
# Return the dataframe
Your output:
# Given the list_of_lists，  header， row ， convert list to dataframe and return it。
Example 3:
User input:
# I need to remain the rows where line_num is not equal to 0.  What's the most efficient way to do it?
# it should be as simple as:
Your output:
# Remain the rows where line_num is not equal to 0 by the most efficient way.
Next, I will provide you with some annotations. You need to follow my requirements and examples to unify the annotation style into a non colloquial and formal style.

Here is the comment which you need handle：
<description for whole class>
import logging
import datetime


class AccessGatewayFilter:

    def __init__(self):
        pass

    def filter(self, request):
        <description for whole function>
        <description for all parameters>
        <description for return statement>
        <some test cases for the function>
        request_uri = request['path']
        method = request['method']

        if self.is_start_with(request_uri):
            return True

        try:
            token = self.get_jwt_user(request)
            user = token['user']
            if user['level'] > 2:
                self.set_current_user_info_and_log(user)
                return True
        except:
            return False

    def is_start_with(self, request_uri):
        <description for whole function>
        <description for all parameters>
        <description for return statement>
        <some test cases for the function>
        start_with = ["/api", '/login']
        for s in start_with:
            if request_uri.startswith(s):
                return True
        return False

    def get_jwt_user(self, request):
        <description for whole function>
        <description for all parameters>
        <description for return statement>
        <some test cases for the function>
        token = request['headers']['Authorization']
        user = token['user']
        if token['jwt'].startswith(user['name']):
            jwt_str_date = token['jwt'].split(user['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return token

    def set_current_user_info_and_log(self, user):
        <description for whole function>
        <description for all parameters>
        <description for return statement>
        <some test cases for the function>
        host = user['address']
        logging.log(msg=user['name'] + host + str(datetime.datetime.now()), level=1)
"""

messages = [
    {"role": "user", "content": prompt},
]

outputs = pipeline(
    messages,
    max_new_tokens=5096,
)
print(outputs[0]["generated_text"][-1])
