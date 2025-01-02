import torch
from sentence_transformers import SentenceTransformer


class SBERTScore:
    def __init__(self, model_name="all-mpnet-base-v2", device="cpu"):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def score_two_questions(self, question_ones, question_twos, lengths=None, extra=None):
        """
        Returns the cosine similarity between questions using SBERT embeddings
        """
        embeddings_one = self.model.encode(question_ones, convert_to_tensor=True, device=self.device)
        embeddings_two = self.model.encode(question_twos, convert_to_tensor=True, device=self.device)
        cosine_similarities = self.cos(embeddings_one, embeddings_two)
        return cosine_similarities.cpu().numpy().tolist()

if __name__ == "__main__":
    s = SBERTScore()
    s.score_two_questions("hello", "hello world")