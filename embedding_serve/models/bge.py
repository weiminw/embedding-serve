from embedding_serve.engine import EmbeddingModel


class BgeM3(EmbeddingModel):
    def text_sparse_embed(self, batch_sentences: list[str]) -> list[dict[int, float]]:
        pass