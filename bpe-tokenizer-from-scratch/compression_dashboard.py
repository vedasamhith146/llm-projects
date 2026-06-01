def compression_dashboard(text,encoded_tokens):
    original_bytes=len(text.encode('utf-8'))
    original_chars=len(text)
    num_tokens=len(encoded_tokens)

    return {
        'original_bytes': original_bytes,
        'token_count': num_tokens,
        'bytes_per_token': original_bytes / num_tokens,
        'characters_per_token': original_chars / num_tokens,
        'unknown_token_rate': sum(1 for t in encoded_tokens if t<256)/num_tokens
    }
