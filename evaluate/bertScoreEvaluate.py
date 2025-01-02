from bert_score import score


class BertScoreEvaluate:
    def __init__(self):
        pass

    def get_score(self, text_1_list, text_2_list):
        # assert type(text_2_list) == list and type(text_1_list) == list, ValueError
        text_1_list = [text_1_list]
        text_2_list = [text_2_list]
        P, R, F1 = score(text_1_list, text_2_list, lang="en", verbose=True, model_type='bert-large-uncased')
        P, R, F1 = map(float, [P, R, F1])
        return P, R, F1


if __name__ == '__main__':
    # Test
    bert_score = BertScoreEvaluate()
    text_1_list = ["My name is Lucky."]
    text_2_list = ["Luck is my dog."]
    # >>> (0.5719687938690186, 0.5650575160980225, 0.5684921741485596)
    print(bert_score.get_score(text_1_list, text_2_list))

