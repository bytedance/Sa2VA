def decode_tokens_with_counts(token_ids, tokenizer):
    """
    Decode tokens and compress consecutive identical tokens with count notation.
    E.g., [1, 1, 1, 2, 2] -> "<token1>x3 <token2>x2"
    """
    if not token_ids:
        return ""
    
    # Decode tokens individually to handle special tokens properly
    tokens = []
    for token_id in token_ids:
        try:
            token = tokenizer.decode([token_id], skip_special_tokens=False)
            tokens.append((token_id, token))
        except:
            tokens.append((token_id, f"<UNK_{token_id}>"))
    
    # Compress consecutive identical tokens
    result = []
    current_token = None
    current_count = 0
    
    for token_id, token_str in tokens:
        if current_token is None:
            current_token = (token_id, token_str)
            current_count = 1
        elif current_token[0] == token_id:
            current_count += 1
        else:
            # Add the previous token(s) to result
            if current_count == 1:
                result.append(current_token[1])
            else:
                result.append(f"{current_token[1]}x{current_count}")
            
            # Start new token
            current_token = (token_id, token_str)
            current_count = 1
    
    # Add the last token(s)
    if current_token is not None:
        if current_count == 1:
            result.append(current_token[1])
        else:
            result.append(f"{current_token[1]}x{current_count}")
    
    return result