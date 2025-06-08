import torch
import torch.nn as nn


# sample text

text = ["I", "am", "a", "sentence", "."]

# assume each token is embedded into a vector with size embed_dim
embed_dim = 5
token_embeds = torch.rand(len(text), 1, embed_dim) # (seq_len, batch_size, embed_dim)

# assume each token has a position embedding
pos_embeds = torch.rand(len(text), 1, embed_dim) # (seq_len, batch_size, embed_dim)

# concatenate the token embeddings and position embeddings

embeds = torch.cat((token_embeds, pos_embeds), dim=2) # (seq_len, batch_size, embed_dim * 2)

# Define Query, Key, and Value Projection

query_proj = nn.Linear(2*embed_dim, 2*embed_dim)
key_proj = nn.Linear(2*embed_dim, 2*embed_dim)
value_proj = nn.Linear(2*embed_dim, 2*embed_dim)

# Generate Query, Key, and Value
query = query_proj(embeds) # (seq_len, batch_size, embed_dim * 2)
key = key_proj(embeds)
value = value_proj(embeds)

print(query.shape)
print(key.transpose(1, 2).shape)

# Compute attention scores

scores = torch.bmm(query, key.transpose(1, 2)) # (seq_len, seq_len, batch_size)
print(scores)
scaled_scores = scores / ((2*embed_dim) ** 0.5) # (seq_len, seq_len, batch_size)

# apply softmax to get attention weights
attention_weights = torch.softmax(scaled_scores, dim=-1) # (seq_len, seq_len, batch_size)

print(attention_weights)

# compute weighted sum of values

output = torch.bmm(attention_weights, value) # (seq_len, seq_len, batch_size)   
