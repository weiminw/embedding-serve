# embedding-serve
embedding-serve is a fast and easy-to-use library for document embedding.  
embedding-serve supports the following:
- [x] text dense embedding: BGE, GTE(Alibaba-NLP)
- [x] text sparse embedding: BGE, GTE(Alibaba-NLP)
- [x] image embedding: SigLIP, ViT

## Getting Started
### Deploying with Docker
```bash
docker run --gpus all -itd -p 8000:8000 /path/to/your/models:/models/ weiminw/heliumos-bixi-embeddings:0.6.0 sh -c "source /workspace/heliumos-env/bin/activate && python -m bixi.embeddings.serve --model /models/model_path --log-level INFO"
```