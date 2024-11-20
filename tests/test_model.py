from bixi.embeddings.models.base import EmbeddingModel

# model = EmbeddingModel("/workspace/models/Baai/bge-m3", use_fp16=True, max_token_length=8192, )
model = EmbeddingModel("/workspace/models/gte-multilingual-base", use_fp16=True, max_token_length=8192, )
# model = EmbeddingModel("/workspace/models/gte-Qwen2-1.5B-instruct", use_fp16=True, max_token_length=8192, )
# embeddings = model.encode(texts=["hello world", "ok", "this is world"])
# print(embeddings)
embeddings = model.encode(texts=[",ltd"], embedding_type='text_sparse_embedding')
embeddings_2 = model.encode(texts=["crwwwoss border traddfdfdfed integgrated servicess cosssss., lsssstd"], embedding_type='text_sparse_embedding')
print(embeddings)
print(embeddings_2)
score= model.compute_sparse_scores(embeddings, embeddings_2)
print(score)
