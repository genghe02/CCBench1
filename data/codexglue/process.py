import sys

from openai import OpenAI
from tqdm import tqdm


from models import deepseek_coder_v2
from run_code import save_json


def deepseek_v2_handle(api_key, base_url, model_name, prompt):
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

def str_delete(target_str: str):
    while '#' in target_str or '\n' in target_str:
        target_str = target_str.replace('\n', '')
        target_str = target_str.replace('#', '')
    return target_str.strip()


prompts = []
ds = open(f"test.jsonl", 'r')


lines = ds.readlines()

with open("test.txt", 'r', encoding='utf-8') as f_1:

    test_lines = f_1.readlines()

# ans = ''
# print(test_lines[0])

# print(eval(lines[0])['url'])

# print(test_lines)
# sys.exit()

ans = []

for i, item in enumerate(tqdm(lines)):
    # js = eval(item)
    # # print(js['url'])
    # # sys.exit()
    # if js['url'] + '\n' not in test_lines:
    #     continue
    # ans += item
    # print(1)

    prompt = """
Task:  You need to generate a summary based on the code content.
    
Attention: 1. Do not include any code content in the reply.
2. The reply only needs to include a summary of the code you generated. Do not include any other content.
3. Do not include useless information such as parameter types and test samples in the code summary.
4. The abstract must be concise, using a concise sentence to respond. For details, please refer to the format of the "Correct Output" output in the following example.
    
    Example 1：
    Input:
    def sina_xml_to_url_list ( xml_data ) : rawurl = [ ] dom = parseString ( xml_data ) for node in dom . getElementsByTagName ( 'durl' ) : url = node . getElementsByTagName ( 'url' ) [ 0 ] rawurl . append ( url . childNodes [ 0 ] . data ) return rawurl
    ❌ Wrong Output (Contains code information):
    def sina_xml_to_url_list ( xml_data ) : # Convert XML to URL List . rawurl = [ ] dom = parseString ( xml_data ) for node in dom . getElementsByTagName ( 'durl' ) : url = node . getElementsByTagName ( 'url' ) [ 0 ] rawurl . append ( url . childNodes [ 0 ] . data ) return rawurl
    ✅ Correct Output:
    Convert XML to URL List .
    
    Example2:
    Input:def ckplayer_get_info_by_xml ( ckinfo ) : e = ET . XML ( ckinfo ) video_dict = { 'title' : '' , #'duration': 0, 'links' : [ ] , 'size' : 0 , 'flashvars' : '' , } dictified = dictify ( e ) [ 'ckplayer' ] if 'info' in dictified : if '_text' in dictified [ 'info' ] [ 0 ] [ 'title' ] [ 0 ] : #title video_dict [ 'title' ] = dictified [ 'info' ] [ 0 ] [ 'title' ] [ 0 ] [ '_text' ] . strip ( ) #if dictify(e)['ckplayer']['info'][0]['title'][0]['_text'].strip(): #duration #video_dict['title'] = dictify(e)['ckplayer']['info'][0]['title'][0]['_text'].strip() if '_text' in dictified [ 'video' ] [ 0 ] [ 'size' ] [ 0 ] : #size exists for 1 piece video_dict [ 'size' ] = sum ( [ int ( i [ 'size' ] [ 0 ] [ '_text' ] ) for i in dictified [ 'video' ] ] ) if '_text' in dictified [ 'video' ] [ 0 ] [ 'file' ] [ 0 ] : #link exist video_dict [ 'links' ] = [ i [ 'file' ] [ 0 ] [ '_text' ] . strip ( ) for i in dictified [ 'video' ] ] if '_text' in dictified [ 'flashvars' ] [ 0 ] : video_dict [ 'flashvars' ] = dictified [ 'flashvars' ] [ 0 ] [ '_text' ] . strip ( ) return video_dict
    ❌ Wrong Output (Contains redundant test samples, parameters, and other information):
    str - > dict Information for CKPlayer API content .
    ✅ Correct Output:Information for CKPlayer API content .
    Information for CKPlayer API content.
    
    Here is the end of examples.
    
    Here is the task you should finish.
    
    Input:
    """

    js = eval(item)
    code = ' '.join(js['code_tokens']).replace('\n', ' ')
    code = ' '.join(code.strip().split())

    prompt = prompt + '\n' + code

    response_text = deepseek_v2_handle(deepseek_coder_v2.Deepseek_v2.api_key,
                                       deepseek_coder_v2.Deepseek_v2.base_url,
                                       deepseek_coder_v2.Deepseek_v2.model_name,
                                       prompt)
    ans.append(response_text)

    save_json("deepseek_v2.json", ans)
# with open("test.jsonl", 'w', encoding='utf-8') as f_2:
#     f_2.write(ans)