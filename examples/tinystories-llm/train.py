import argparse
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torchmlx as torch
from torchmlx import nn, optim
import torchmlx.nn.functional as F


device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DATA_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
DATA_PATH = Path(__file__).with_name("TinyStoriesV2-GPT4-valid.txt")


@dataclass
class TransformerConfig:
    vocabulary_size: int
    context_length: int = 64
    model_dimension: int = 64
    head_count: int = 4
    layer_count: int = 2
    feed_forward_dimension: int = 256


class CharacterTokenizer:
    def __init__(self, text):
        self.characters = sorted(set(text))
        self.token_by_character = {
            character: token for token, character in enumerate(self.characters)
        }

    def encode(self, text):
        return [self.token_by_character[character] for character in text]

    def decode(self, tokens):
        return "".join(self.characters[int(token)] for token in tokens)

    def __len__(self):
        return len(self.characters)


class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocabulary_size, config.model_dimension
        )

    def forward(self, tokens):
        return self.embedding(tokens)


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.context_length, config.model_dimension
        )

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.embedding(positions)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_count = config.head_count
        self.head_dimension = config.model_dimension // config.head_count
        self.query = nn.Linear(config.model_dimension, config.model_dimension)
        self.key = nn.Linear(config.model_dimension, config.model_dimension)
        self.value = nn.Linear(config.model_dimension, config.model_dimension)
        self.output = nn.Linear(config.model_dimension, config.model_dimension)

    def split_heads(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, sequence_length, self.head_count, self.head_dimension
        )
        return hidden_states.transpose(1, 2)

    def merge_heads(self, hidden_states):
        batch_size, _, sequence_length, _ = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        return hidden_states.reshape(
            batch_size, sequence_length, self.head_count * self.head_dimension
        )

    def forward(self, hidden_states):
        queries = self.split_heads(self.query(hidden_states))
        keys = self.split_heads(self.key(hidden_states))
        values = self.split_heads(self.value(hidden_states))

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / self.head_dimension**0.5

        sequence_length = hidden_states.shape[1]
        future_positions = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(
            future_positions, float("-inf")
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, values)
        return self.output(self.merge_heads(attended_values))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.model_dimension, config.feed_forward_dimension),
            nn.GELU(),
            nn.Linear(config.feed_forward_dimension, config.model_dimension),
        )

    def forward(self, hidden_states):
        return self.layers(hidden_states)


class Unembedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output = nn.Linear(
            config.model_dimension, config.vocabulary_size, bias=False
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.model_dimension)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.model_dimension)
        self.feed_forward = FeedForward(config)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states)
        )
        return hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )


class TinyStoriesTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.layer_count)]
        )
        self.final_norm = nn.LayerNorm(config.model_dimension)
        self.unembedding = Unembedding(config)

    def forward(self, tokens):
        hidden_states = self.token_embedding(tokens)
        hidden_states = hidden_states + self.position_embedding(tokens)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.unembedding(self.final_norm(hidden_states))


def load_tiny_stories():
    if not DATA_PATH.exists():
        print(f"downloading TinyStories to {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def create_batch(encoded_text, batch_size, context_length):
    starts = [
        random.randrange(len(encoded_text) - context_length - 1)
        for _ in range(batch_size)
    ]
    input_tokens = [
        encoded_text[start : start + context_length] for start in starts
    ]
    target_tokens = [
        encoded_text[start + 1 : start + context_length + 1]
        for start in starts
    ]
    return torch.tensor(input_tokens, dtype=torch.long, device=device), torch.tensor(
        target_tokens, dtype=torch.long, device=device
    )


def language_model_loss(logits, target_tokens):
    vocabulary_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocabulary_size), target_tokens.reshape(-1)
    )


def generate_text(model, tokenizer, prompt, token_count):
    generated_tokens = tokenizer.encode(prompt)
    model.eval()
    for _ in range(token_count):
        context = generated_tokens[-model.config.context_length :]
        input_tokens = torch.tensor([context], dtype=torch.long, device=device)
        next_token_logits = model(input_tokens)[:, -1, :]
        next_token = torch.categorical(next_token_logits).item()
        generated_tokens.append(next_token)
    return tokenizer.decode(generated_tokens)


parser = argparse.ArgumentParser()
parser.add_argument("steps", nargs="?", type=int, default=20)
arguments = parser.parse_args()

training_text = load_tiny_stories()
tokenizer = CharacterTokenizer(training_text)
config = TransformerConfig(vocabulary_size=len(tokenizer))
encoded_text = tokenizer.encode(training_text)
model = TinyStoriesTransformer(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
model.train()

for step in range(arguments.steps):
    input_tokens, target_tokens = create_batch(
        encoded_text, batch_size=8, context_length=config.context_length
    )
    optimizer.zero_grad()
    logits = model(input_tokens)
    loss = language_model_loss(logits, target_tokens)
    loss.backward()
    optimizer.step()
    print(f"step {step + 1}: loss {loss.item():.4f}")

print(generate_text(model, tokenizer, "Once upon a time", token_count=120))
