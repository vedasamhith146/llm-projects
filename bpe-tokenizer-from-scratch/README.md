# Project 1: Build a Tokenizer from Scratch

This is the first project in the series of LLM engineering projects: **Project 1 - Build a Tokenizer from Scratch**.

In this project, I implemented the **Byte Pair Encoding (BPE)** algorithm and trained a tokenizer on the Tiny Shakespeare dataset. I implemented four different versions of the tokenizer:

- Version 1: 100 merges (vocabulary size = 356)
- Version 2: 500 merges
- Version 3: 1500 merges
- Version 4: 2500 merges

For each version, I calculated:

- Characters per token
- Bytes per token
- Unknown token rate

I also analyzed token counts across different domains. Since the tokenizer was trained on English text, I evaluated its performance on multilingual text, code, URLs, and other domains.

Additionally, I tested how:

- A small vocabulary increases sequence length.
- A large vocabulary can under-train rare tokens.

I compared the tokenization of the following text at different stages (100, 500, 1500, and 2500 merges):

> "Understanding natural language processing requires careful attention to tokenization strategies. The algorithm quickly learns frequent patterns, but rare technical terms often remain fragmented until more merges are applied."

### After 100 Merges

```python
{
    'original_bytes': 224,
    'token_count': 157,
    'bytes_per_token': 1.4267515923566878,
    'characters_per_token': 1.4267515923566878,
    'unknown_token_rate': 0.6050955414012739
}
```

### After 500 Merges

```python
{
    'original_bytes': 224,
    'token_count': 126,
    'bytes_per_token': 1.7777777777777777,
    'characters_per_token': 1.7777777777777777,
    'unknown_token_rate': 0.35714285714285715
}
```

### After 1500 Merges

```python
{
    'original_bytes': 224,
    'token_count': 117,
    'bytes_per_token': 1.9145299145299146,
    'characters_per_token': 1.9145299145299146,
    'unknown_token_rate': 0.27350427350427353
}
```

### After 2500 Merges

```python
{
    'original_bytes': 224,
    'token_count': 110,
    'bytes_per_token': 2.036363636363636,
    'characters_per_token': 2.036363636363636,
    'unknown_token_rate': 0.21818181818181817
}
```

I also used a Shakespeare-like text with the same number of original bytes as the previous text and observed a lower token count. This suggests that when the training corpus is more similar to the input text, fewer tokens are required.

```python
text_shakespeare = "What light through yonder window breaks? 'Tis the east, and Juliet is the sun. Arise, fair sun, and kill the envious moon that doth infect this weary world with grief and sorrow. Yet hope remains in hearts that love true and"
```

```python
{
    'original_bytes': 224,
    'token_count': 101,
    'bytes_per_token': 2.217821782178218,
    'characters_per_token': 2.217821782178218,
    'unknown_token_rate': 0.19801980198019803
}
```

## Domain Testing

I tested my tokenizer on the following domains:

```python
test_domains = {
    "english_prose": "The quick brown fox jumps over the lazy dog. It was the best of times, it was the worst of times.",
    "code_python": "def train_model(data, epochs=10): return model.fit(data, verbose=True)",
    "url_path": "https://api.example.com/v2/users?id=42&format=json&token=abc123",
    "math_notation": "E = mc^2, ∫f(x)dx, ∀x∈ℝ: x² ≥ 0, lim_{n→} (1 + 1/n)^n = e",
    "social_media": "🚀 Just launched! #AI #MachineLearning @OpenAI check https://t.co/xyz 🔥🤖",
    "technical_words": "Photosynthesis, mitochondria, electrocardiogram, pneumonoultramicroscopicsilicovolcanoconiosis",
    "json_data": '{"status": 200, "message": "success", "data": [{"id": 1, "value": true}]}',

    "arabic": "مرحبا بالعالم! كيف حالك اليوم؟ هذا اختبار للتوكنيزر.",
    "hindi": "नमस्ते दुनिया! आप कैसे हैं? यह टोकेनाइज़र का परीक्षण है।",
    "chinese": "你好世界！今天天气怎么样？这是一个分词器测试。",
    "emojis": "🌍🚀🔥✨👨‍👩‍👦🎉💻🤖🧠",
    "corrupted_unicode": "Valid text \ufffd mixed with \ufffd invalid sequences \ufffd and replacement chars."
}
```

The following results were recorded:

## Domain Testing Results

| Domain | Original Bytes | Tokenized Bytes | Compression Ratio | Bytes Per Token |
|---------|---------------:|----------------:|------------------:|----------------:|
| English Prose | 97 | 46 | 2.11× | 2.1087 |
| Python Code | 70 | 46 | 1.52× | 1.5217 |
| URL Path | 63 | 46 | 1.37× | 1.3696 |
| Mathematical Notation | 70 | 63 | 1.11× | 1.1111 |
| Social Media | 80 | 58 | 1.38× | 1.3793 |
| Technical Words | 94 | 51 | 1.84× | 1.8431 |
| JSON Data | 73 | 49 | 1.49× | 1.4898 |
| Arabic | 95 | 94 | 1.01× | 1.0106 |
| Hindi | 146 | 144 | 1.01× | 1.0139 |
| Chinese | 69 | 69 | 1.00× | 1.0000 |
| Emojis | 52 | 52 | 1.00× | 1.0000 |
| Corrupted Unicode | 74 | 42 | 1.76× | 1.7619 |