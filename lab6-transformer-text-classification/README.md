# Lab 6: Transformer for Text Classification

Implements a Transformer encoder from scratch (positional encoding, multi-head self-attention, feed-forward blocks) and applies it to text classification on the AG News dataset.

## Contents (`lab6_transformer_text_classification.ipynb`)

### Architecture (built from scratch, not `nn.Transformer`)
- **`PositionalEncoding`** — sinusoidal position embeddings added to token embeddings
- **`AttentionMulti`** — multi-head self-attention, with Q/K/V computed via a single fused linear projection (`w_qkv`)
- **`Transformer`** — stack of alternating attention + MLP blocks (configurable depth)
- **`TextClassifyTransformer`** — wraps the Transformer encoder with token embedding + classification head for 4-way text classification

### Data
- **Dataset:** AG News (4-class news topic classification), loaded via Hugging Face `datasets.load_dataset("ag_news")`
- **Tokenizer:** `bert-base-uncased` (`BertTokenizerFast`) — used only for tokenization/ID mapping, not as a pretrained embedding source
- **Sequence length:** 512 (truncated/padded)

### Training Setup
- **Model config:** depth=6, embedding dim=300, attention dim=128, heads=4, MLP dim=1200, dropout=0.1
- **Epochs:** 5, **Batch size:** 16, **Learning rate:** 1e-5
- **Optimizer:** AdamW (weight decay 1e-2), with `StepLR` scheduling (gamma=0.7 per epoch)
- **Loss:** CrossEntropyLoss
- **Logging:** TensorBoard (`SummaryWriter`, logs written to `./log/`)

## Extension Explored
The assignment references *Gated Attention for Large Language Models* (NeurIPS 2025 Best Paper) and its proposed fix for Transformers: inserting a sigmoid gate between the attention value projection and the output projection. This introduces non-linearity into an otherwise linear value→output path, gives the model a form of "selective silence," and helps eliminate the attention-sink phenomenon. The notebook applies this gating idea to the text classification model above.

## Tech Stack
- Python, PyTorch
- Hugging Face `transformers` (tokenizer), `datasets` (AG News)
- `einops` (tensor reshaping), TensorBoard (logging)

## Requirements
```bash
pip install torch datasets transformers einops tensorboard
```

## Usage
Run all cells in `lab6_transformer_text_classification.ipynb` sequentially. GPU is used automatically if available. Training/validation metrics are printed per epoch and logged to TensorBoard.

## Notes
- Several sections (positional encoding, multi-head attention) were originally fill-in-the-blank exercises, completed as part of this assignment.