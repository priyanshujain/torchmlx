import argparse
import random
import urllib.request
from pathlib import Path

import torchmlx as torch
from torchmlx import nn, optim
import torchmlx.nn.functional as F


DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
DATA_PATH = Path(__file__).with_name("TinyStoriesV2-GPT4-valid.txt")


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        q = torch.transpose(
            q.reshape(batch, length, self.heads, self.head_width), 1, 2
        )
        k = torch.transpose(
            k.reshape(batch, length, self.heads, self.head_width), 1, 2
        )
        v = torch.transpose(
            v.reshape(batch, length, self.heads, self.head_width), 1, 2
        )
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = torch.transpose(x, 1, 2).reshape(batch, length, width)
        return self.output(x)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feed_forward_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class TinyStoriesModel(nn.Module):
    def __init__(self, vocab_size, context, width=64, heads=4, layers=2):
        super().__init__()
        self.context = context
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


def download_data():
    if not DATA_PATH.exists():
        print(f"downloading TinyStories to {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def batch(encoded, batch_size, context):
    starts = [random.randrange(len(encoded) - context - 1) for _ in range(batch_size)]
    x = [encoded[start : start + context] for start in starts]
    y = [encoded[start + 1 : start + context + 1] for start in starts]
    return torch.tensor(x, dtype=torch.int64, device=device), torch.tensor(
        y, dtype=torch.int64, device=device
    )


def generate(model, prompt, encode, decode, length):
    tokens = encode(prompt)
    for _ in range(length):
        x = torch.tensor([tokens[-model.context :]], dtype=torch.int64, device=device)
        logits = model(x)[:, -1, :]
        token = torch.categorical(logits).item()
        tokens.append(token)
    return decode(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="?", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    text = download_data()
    characters = sorted(set(text))
    to_id = {character: index for index, character in enumerate(characters)}
    encode = lambda value: [to_id[character] for character in value]
    decode = lambda values: "".join(characters[int(value)] for value in values)
    encoded = encode(text)
    context = 64
    model = TinyStoriesModel(len(characters), context).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = lambda logits, targets: F.cross_entropy(
        logits.reshape(-1, len(characters)), targets.reshape(-1)
    )
    trainer = torch.Trainer(
        model, optimizer, loss_fn, compile=not args.no_compile
    )
    interval = max(1, args.steps // 10)
    for step in range(args.steps):
        x, y = batch(encoded, 8, context)
        loss = trainer.step(x, y)
        if step % interval == 0 or step == args.steps - 1:
            print(f"step {step + 1}: loss {loss.item():.4f}")
    print(generate(model, "Once upon a time", encode, decode, 120))


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if __name__ == "__main__":
    main()
