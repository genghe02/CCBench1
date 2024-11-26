from pyrouge import Rouge155


class RougeEvaluate:
    def __init__(self):
        self.r = Rouge155()

    def get_score(self, reference, hypothesis):
        """
        :param reference:
        :param hypothesis:
        :return: Dict of scores
        Keys of return dict:
        ['rouge_1_recall', 'rouge_1_recall_cb', 'rouge_1_recall_ce',
        'rouge_1_precision', 'rouge_1_precision_cb', 'rouge_1_precision_ce',
         'rouge_1_f_score', 'rouge_1_f_score_cb', 'rouge_1_f_score_ce',
         'rouge_2_recall', 'rouge_2_recall_cb', 'rouge_2_recall_ce',
         'rouge_2_precision', 'rouge_2_precision_cb', 'rouge_2_precision_ce',
         'rouge_2_f_score', 'rouge_2_f_score_cb', 'rouge_2_f_score_ce',
         'rouge_3_recall', 'rouge_3_recall_cb', 'rouge_3_recall_ce',
         'rouge_3_precision', 'rouge_3_precision_cb', 'rouge_3_precision_ce',
         'rouge_3_f_score', 'rouge_3_f_score_cb', 'rouge_3_f_score_ce',
         'rouge_4_recall', 'rouge_4_recall_cb', 'rouge_4_recall_ce',
         'rouge_4_precision', 'rouge_4_precision_cb', 'rouge_4_precision_ce',
         'rouge_4_f_score', 'rouge_4_f_score_cb', 'rouge_4_f_score_ce',
         'rouge_l_recall', 'rouge_l_recall_cb', 'rouge_l_recall_ce',
         'rouge_l_precision', 'rouge_l_precision_cb', 'rouge_l_precision_ce',
         'rouge_l_f_score', 'rouge_l_f_score_cb', 'rouge_l_f_score_ce',
         'rouge_w_1.2_recall', 'rouge_w_1.2_recall_cb', 'rouge_w_1.2_recall_ce',
         'rouge_w_1.2_precision', 'rouge_w_1.2_precision_cb', 'rouge_w_1.2_precision_ce',
         'rouge_w_1.2_f_score', 'rouge_w_1.2_f_score_cb', 'rouge_w_1.2_f_score_ce',
         'rouge_s*_recall', 'rouge_s*_recall_cb', 'rouge_s*_recall_ce',
         'rouge_s*_precision', 'rouge_s*_precision_cb', 'rouge_s*_precision_ce',
         'rouge_s*_f_score', 'rouge_s*_f_score_cb', 'rouge_s*_f_score_ce',
         'rouge_su*_recall', 'rouge_su*_recall_cb', 'rouge_su*_recall_ce',
         'rouge_su*_precision', 'rouge_su*_precision_cb', 'rouge_su*_precision_ce',
         'rouge_su*_f_score', 'rouge_su*_f_score_cb', 'rouge_su*_f_score_ce']
        """
        with open("/Users/genghe/PycharmProjects/CCBench/pyrouge/test1/1.txt", 'w') as f:
            f.write(reference)
        with open("/Users/genghe/PycharmProjects/CCBench/pyrouge/test2/2.txt", 'w') as f:
            f.write(hypothesis)
        self.r.system_dir = '/Users/genghe/PycharmProjects/CCBench/pyrouge/test1'  # 'path/to/system_summaries'
        self.r.model_dir = '/Users/genghe/PycharmProjects/CCBench/pyrouge/test2'  # 'path/to/model_summaries'
        self.r.system_filename_pattern = '^[\w,\s-]+\.(txt)$'
        self.r.model_filename_pattern = '^[\w,\s-]+\.(txt)$'  # 所以用正则表达式来匹配了数字0001.txt这种
        score = self.r.convert_and_evaluate()
        output_dict = self.r.output_to_dict(score)
        return output_dict


if __name__ == '__main__':
    # Test case
    rouge = RougeEvaluate()
    rouge.get_score("Hello, nice to meet you.", "Hi, glad to see you.")
