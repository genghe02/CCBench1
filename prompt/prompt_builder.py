# classEval prompt
import sys
sys.path.append("..")
from tools.json_open_close import open_json, save_json
class PromptBuilder:
    """
    Build prompt
    """

    def __init__(self, prompt_type='classeval'):
        assert prompt_type in ['classeval']
        self.class_eval_data = open_json("../data/ClassEval-master/data/ClassEval_data.json")

    @staticmethod
    def add_desc_to_init(desc, class_init):
        class_init_list = class_init.split('\n')
        class_init_list[0] += " \n" + desc
        class_init = '\n'.join(class_init_list)
        return class_init

    @staticmethod
    def get_method_signature(code, method_name):
        method_def_prefix = "def " + method_name + '('
        code_segment = code.split('):')
        for segment in code_segment:
            if method_def_prefix in segment:
                return "    " + segment + "):"
        return ""

    def construct_prompt(self, strategy, info):
        prompt = ""
        if strategy == 0:
            class_name = info['class_name']
            skeleton = info['skeleton']
            instruction = f"Please complete the class {class_name} in the following code."
            instruction = instruction + '\n' + skeleton
            prompt = self.generate_classeval_code_generation_prompt(instruction)

        elif strategy == 1:
            prompt = info['instruction'] + info['skeleton']
            prompt = self.generate_classeval_code_generation_prompt(prompt)

        elif strategy == 2:
            prompt = info['instruction'] + info['skeleton']
            prompt = self.generate_classeval_code_generation_prompt(prompt)

        return prompt

    def generate_single_code_generation_prompt(self):
        cont = self.class_eval_data[0]
        cont['predict'] = []
        error_task_id_list = []
        print(cont['class_name'])
        # try:
        class_name = cont['class_name']
        methods_info = cont['methods_info']
        imports = '\n'.join(cont['import_statement'])
        class_init = self.add_desc_to_init(cont['class_description'], cont['class_constructor'])
        for method_to_generate in methods_info:
            class_text = imports + '\n' + class_init
            # gather each method's signature to consruct class level skeleton
            for method in methods_info:
                if method['method_name'] == method_to_generate['method_name']:
                    continue
                class_text += self.get_method_signature(method['method_description'],
                                                                 method['method_name']) + "\n        pass\n\n"
            # construct prompt
            method_name = method_to_generate['method_name']
            inst = f"please complete {method_name} method in the following class {class_name}\n\n"
            class_text_desc = class_text + "\n\n    " + method_to_generate['method_description']
            prompt = self.construct_prompt(1,
                                           {"instruction": inst, "skeleton": class_text_desc})
            print(prompt)
            print('==================')
        # cont['predict'].append(pred)

        # except Exception as e:
        #     print(e)
        #     print("IDX: ", cont['task_id'])
        #     error_task_id_list.append(cont['task_id'])
        # return prompt


    def generate_classeval_code_generation_prompt(self, instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
                # 
                # ### Instruction:
                # {instruction}
                # 
                # ### Response:
                # """
        # if model_name == ModelName.DeepSeekCoder_inst.value or model_name == ModelName.Gemini_Pro.value:
        #     return instruction
        #
        # elif model_name == ModelName.Magicoder.value:
        #     return f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
        #
        # @@ Instruction:
        # {instruction}
        #
        # @@ Response:
        # """
        # else:
        #     return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        #
        # ### Instruction:
        # {instruction}
        #
        # ### Response:
        # """

    def generate_comments_prompt(self):
        class_description = '<description for whole class>'
        function_description = '<description for whole function>'
        param_description = '<description for all parameters>'
        return_description = '<description for return statement>'
        test_case = '<some test cases for the function>'


if __name__ == '__main__':
    prompt_builder = PromptBuilder()
    # prompt_builder.generate_classeval_code_generation_prompt()
    prompt_builder.generate_single_code_generation_prompt()