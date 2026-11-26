# Understanding Self-Attention in Transformer Architecture

## Introduction to Transformer Architecture and Self-Attention

The Transformer architecture is a groundbreaking neural network design that has revolutionized natural language processing (NLP) and extended its impact to other domains such as computer vision and speech recognition. Unlike traditional sequence models based on recurrent neural networks (RNNs) or convolutional neural networks (CNNs), Transformers rely entirely on attention mechanisms to process input data. This shift enables more efficient and scalable modeling of complex dependencies within sequences.

The motivation for using self-attention over recurrent or convolutional layers stems from the limitations inherent in those architectures. RNNs process sequences sequentially, which hinders parallelization and makes it difficult to capture long-range dependencies due to vanishing gradients. CNNs, while parallelizable, have a fixed receptive field size that limits their ability to model relationships between distant tokens. Self-attention addresses these issues by allowing every position in the input sequence to directly attend to every other position, regardless of their distance.

Self-attention is a mechanism that computes a weighted representation of a sequence by relating different positions within that same sequence. For each token, it generates a context-aware embedding by aggregating information from all tokens, weighted by learned attention scores. This enables the model to dynamically focus on relevant parts of the sequence when encoding or decoding, capturing nuanced dependencies and contextual relationships.

In the Transformer architecture, data flows through stacked encoder and decoder layers. Each encoder layer contains a self-attention sub-layer followed by a feed-forward network, while each decoder layer includes masked self-attention, encoder-decoder attention, and feed-forward sub-layers. The self-attention modules in both encoder and decoder allow the model to build rich, context-sensitive representations by attending to all positions in the input or output sequences simultaneously.

Key advantages of the Transformer’s self-attention mechanism include:

- **Parallelism:** Unlike RNNs, self-attention processes all tokens simultaneously, enabling efficient training on modern hardware.
- **Long-range dependency modeling:** It can directly connect distant tokens without the bottleneck of sequential processing.
- **Dynamic context aggregation:** Attention weights adapt based on input, allowing flexible and interpretable focus on relevant sequence parts.

Understanding these fundamentals equips developers to leverage Transformers effectively and optimize their implementations for various sequence modeling tasks.

> **[IMAGE GENERATION FAILED]** Overview of Transformer architecture highlighting encoder and decoder layers with self-attention mechanisms.
>
> **Alt:** Diagram of Transformer architecture showing encoder and decoder stacks with self-attention and feed-forward layers
>
> **Prompt:** A clear, labeled diagram of the Transformer architecture showing stacked encoder and decoder layers, each with self-attention and feed-forward sublayers, illustrating data flow between input, encoder, decoder, and output sequences.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 1h57m54.855507847s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '7074s'}]}}


## Introduction to Transformer Architecture

Transformer models were introduced to address key limitations of traditional sequence models such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs). Unlike these models, which process input sequentially and thus suffer from long training times and difficulty capturing long-range dependencies, Transformers leverage self-attention mechanisms that allow for direct connections between all positions in the input sequence. This design enables more efficient learning of contextual relationships regardless of distance in the sequence.

The Transformer architecture is composed primarily of two parts: the encoder and the decoder. The encoder consists of stacked layers that apply self-attention and feed-forward neural networks to transform the input sequence into a rich representation. The decoder similarly uses self-attention layers but also incorporates encoder-decoder attention to generate output sequences conditioned on the encoded input. Central to both components is the attention mechanism, which dynamically weights the importance of different tokens relative to each other, enabling the model to focus on relevant parts of the sequence during processing.

A significant advantage of the Transformer design is its inherent parallelization capability. Since the self-attention mechanism does not rely on sequential data processing, it allows for simultaneous computation across all tokens in a sequence. This parallelism leads to substantial improvements in training speed and scalability, making Transformers well-suited for large datasets and complex tasks.

Transformers have become foundational in natural language processing (NLP), powering state-of-the-art models for tasks such as machine translation, text summarization, and question answering. Beyond NLP, their architecture has been adapted for applications in computer vision, speech recognition, and even reinforcement learning, demonstrating the versatility and broad impact of the Transformer model.

## Mathematical Formulation of Self-Attention

In Transformer architectures, self-attention is a mechanism that allows each position in the input sequence to attend to all other positions, enabling the model to capture contextual relationships effectively. The core mathematical operations involve three key components derived from the input embeddings: queries (Q), keys (K), and values (V).

### Input Representations: Queries, Keys, and Values

Given an input sequence represented as embeddings \( X \in \mathbb{R}^{n \times d} \), where \( n \) is the sequence length and \( d \) is the embedding dimension, the queries, keys, and values are computed as linear projections of \( X \):

\[
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
\]

Here, \( W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k} \) are learned projection matrices, and \( d_k \) is the dimensionality of the queries and keys (often \( d_k = d / h \) for \( h \) attention heads).

### Scaled Dot-Product Attention

The fundamental operation in self-attention is the scaled dot-product between queries and keys, which measures the compatibility or similarity between elements in the sequence:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

The term \( QK^\top \) produces an \( n \times n \) matrix of raw attention scores, representing how much each position should attend to every other position.

The division by \( \sqrt{d_k} \) is a scaling factor introduced to prevent the dot products from growing too large in magnitude, which can push the softmax function into regions with extremely small gradients, thus stabilizing training.

### Softmax and Attention Weights

The softmax function normalizes the scaled scores row-wise to produce attention weights:

\[
\alpha_{ij} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^n \exp\left(\frac{q_i \cdot k_m}{\sqrt{d_k}}\right)}
\]

where \( \alpha_{ij} \) represents the attention weight from position \( i \) to position \( j \). These weights quantify the influence of each value vector \( v_j \) on the output at position \( i \).

### Output as Weighted Sum of Values

The output of the self-attention layer at each position is a weighted sum of the value vectors, weighted by the attention weights:

\[
\text{Output}_i = \sum_{j=1}^n \alpha_{ij} v_j
\]

This operation aggregates information from the entire sequence, allowing the model to focus on relevant parts dynamically.

### Minimal Code Sketch: Scaled Dot-Product Attention

Below is a minimal PyTorch-style implementation illustrating these computations:

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    # Compute raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # Apply softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)
    # Weighted sum of values
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Example usage with batch size 1, sequence length 3, embedding dimension 4
Q = torch.randn(1, 3, 4)
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print("Output shape:", output.shape)  # Expected: (1, 3, 4)
```

### Debugging Tips

- Verify the dimensions of \( Q, K, V \) to ensure they match expected shapes.
- Confirm that the scaling factor \( \sqrt{d_k} \) is applied before softmax.
- Check that the softmax is applied along the correct dimension (usually the last dimension of the scores matrix).
- Use attention weight visualizations to inspect if the model attends to reasonable positions.

Understanding these mathematical operations is crucial for implementing and debugging Transformer-based models effectively.

> **[IMAGE GENERATION FAILED]** Visualization of the scaled dot-product attention mechanism showing computation of attention scores and weighted sum of values.
>
> **Alt:** Diagram illustrating scaled dot-product attention mechanism with queries, keys, values, and attention weights
>
> **Prompt:** A detailed diagram illustrating the scaled dot-product attention mechanism: input embeddings projected into queries, keys, and values; dot product between queries and keys; scaling by square root of key dimension; softmax to produce attention weights; weighted sum of values to produce output.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 1h57m54.38994913s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '7074s'}]}}


## Conceptual Overview of Self-Attention

Self-attention is a mechanism designed to relate different positions within a single input sequence to compute a contextualized representation for each token. Unlike traditional sequence models that process tokens sequentially, self-attention allows the model to consider all tokens simultaneously, capturing dependencies regardless of their distance in the sequence.

At its core, self-attention dynamically weighs the importance of each token relative to others. For every token in the sequence, the model calculates attention scores that quantify how much focus should be placed on other tokens when encoding that token’s representation. This dynamic weighting enables the model to adaptively emphasize relevant context, improving its ability to model complex relationships such as long-range dependencies and hierarchical structures.

Traditional attention mechanisms typically operate between two distinct sequences, such as in encoder-decoder architectures where the decoder attends to the encoder’s outputs. In contrast, self-attention operates within a single sequence, allowing each token to attend to all tokens including itself. This distinction is fundamental to the Transformer architecture, enabling parallel computation and richer contextualization.

The mechanism relies on three key components for each token: query, key, and value vectors. These vectors are learned linear projections of the token embeddings. The query vector of a token is compared against the key vectors of all tokens to produce attention scores, which are then normalized (usually via softmax). These scores weight the value vectors, resulting in a weighted sum that forms the token’s new representation. This process is repeated for every token, enabling the model to integrate information from the entire sequence effectively.

Understanding these foundational elements is crucial for developers aiming to implement or optimize Transformer models, as they directly influence model capacity, interpretability, and computational efficiency.

## Multi-Head Self-Attention Mechanism

Multi-head self-attention extends the basic self-attention mechanism by running multiple attention operations in parallel, each called an attention head. Instead of computing a single attention output, the model learns to attend to information from different representation subspaces simultaneously. This parallelism enables the Transformer to capture a richer set of features from the input sequence.

Each attention head has its own set of learned linear projection matrices for queries (Q), keys (K), and values (V). Specifically, for an input embedding \(X \in \mathbb{R}^{n \times d}\) (where \(n\) is sequence length and \(d\) is embedding dimension), each head \(i\) applies three linear transformations:

\[
Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V
\]

where \(W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}\) are learned parameters, and \(d_k = d / h\) if there are \(h\) heads. This dimensionality reduction per head keeps the total computation manageable.

After computing scaled dot-product attention for each head independently:

\[
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_k}}\right) V_i
\]

the outputs of all heads are concatenated along the feature dimension:

\[
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\]

Here, \(W^O \in \mathbb{R}^{d \times d}\) is a learned linear projection that mixes the information from all heads back into the original embedding space.

The key benefit of multi-head attention is its ability to capture diverse aspects of the input sequence simultaneously. Different heads can focus on distinct positions or features, such as syntactic relations, semantic roles, or positional patterns. This diversity enhances the model’s expressiveness and robustness compared to a single-head attention mechanism.

### Minimal Code Sketch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Final linear projection
        out = self.out_proj(context)
        return out
```

### Debugging Tips

- Verify that the embedding dimension is divisible by the number of heads to avoid shape mismatches.
- Check tensor shapes after each reshape and transpose operation to ensure correct head splitting and concatenation.
- Use small input sequences and print intermediate attention scores to confirm the softmax outputs are valid probability distributions.
- Compare outputs with a single-head attention implementation to isolate issues in multi-head splitting or projection layers.

By implementing multi-head self-attention as above, developers can leverage the full power of Transformers to model complex dependencies in sequential data.

## Implementation Considerations and Performance Optimizations

When implementing self-attention in Transformer architectures, developers must carefully address computational complexity and memory usage, especially as sequence lengths grow. The standard self-attention mechanism computes pairwise interactions between all tokens, resulting in a time and space complexity of O(N), where N is the sequence length. This quadratic scaling can quickly become a bottleneck for long sequences, leading to excessive memory consumption and slower training or inference times.

To mitigate these challenges, several optimization techniques are commonly employed:

- **Masking:** Applying attention masks allows selective focus on relevant tokens, such as causal masks in autoregressive models or padding masks to ignore padded tokens. This reduces unnecessary computations and ensures correct gradient flow.

- **Batching:** Efficient batching strategies maximize hardware utilization. Grouping sequences of similar lengths minimizes padding overhead, reducing wasted computation and memory.

- **Efficient Matrix Multiplication:** Leveraging optimized linear algebra libraries and fused kernels can significantly speed up the core Q-K-V matrix multiplications. Techniques like mixed precision training also reduce memory bandwidth and improve throughput without sacrificing model accuracy.

Beyond these, approximate or sparse attention methods offer promising avenues to reduce computational cost:

- **Sparse Attention:** By restricting attention to a subset of tokens (e.g., local windows or learned patterns), sparse attention reduces the quadratic complexity to near-linear, enabling scaling to longer sequences.

- **Low-Rank Approximations:** Methods that approximate the full attention matrix with low-rank factors can also decrease memory and compute requirements.

- **Kernel-based Approaches:** Reformulating attention as kernel operations allows linear-time complexity, though these methods may trade off some accuracy.

Hardware acceleration plays a crucial role in practical performance:

- **GPUs:** Their parallel processing capabilities and optimized libraries (e.g., cuBLAS, cuDNN) accelerate matrix operations fundamental to self-attention.

- **TPUs:** Designed for large-scale tensor computations, TPUs offer high throughput and memory bandwidth, particularly beneficial for training large Transformer models.

To ensure robust implementations, developers should watch for common pitfalls:

- **Dimension Mismatches:** Carefully verify tensor shapes at each step, especially during Q, K, V projections and when applying masks. Using assert statements or shape-checking utilities can catch errors early.

- **Numerical Instability:** Softmax operations in attention can suffer from overflow or underflow. Implementing stable softmax variants (e.g., subtracting the max before exponentiation) helps maintain numerical precision.

- **Gradient Issues:** Monitor for exploding or vanishing gradients, which can arise from improper initialization or scaling factors in attention computations.

By combining these optimization strategies and debugging practices, developers can build efficient, scalable self-attention modules that perform well across diverse hardware and sequence lengths.

## Debugging and Observability Tips for Self-Attention Models

Effective debugging and observability are crucial when developing self-attention models to ensure they learn meaningful representations and avoid common pitfalls. Here are practical strategies to help developers monitor and troubleshoot self-attention mechanisms:

- **Visualize Attention Weights:**  
  Visualizing attention matrices can reveal where the model focuses during each layer's computation. Heatmaps or attention maps plotted over input tokens help interpret which parts of the sequence influence the output. This insight is valuable for verifying that the model attends to relevant context rather than irrelevant tokens.

- **Detecting Common Issues:**  
  Watch for signs of vanishing gradients, which can manifest as stagnant or near-zero updates in attention parameters during training. Attention collapsewhere attention weights become uniform or overly concentrated on a few tokenscan also degrade performance. Monitoring gradient norms and attention weight distributions across training steps helps identify these problems early.

- **Tools and Libraries for Visualization:**  
  Libraries such as BertViz, Captum, and TensorBoards embedding projector provide built-in support for attention visualization. These tools enable interactive exploration of attention patterns across layers and heads, facilitating deeper understanding and debugging.

- **Logging Intermediate Outputs and Sanity Checks:**  
  Implement logging of intermediate attention scores and key activations during training. Regularly perform sanity checks, such as verifying that attention weights sum to one and that no NaNs or infinities appear. This practice helps catch numerical instabilities and implementation errors promptly.

- **Testing Robustness with Edge Cases and Adversarial Inputs:**  
  Evaluate the models attention behavior on edge cases, such as very long sequences, repetitive tokens, or inputs with noise. Additionally, adversarial inputs designed to confuse attention can expose vulnerabilities. Observing how attention distributions shift under these conditions informs robustness improvements.

> **[IMAGE GENERATION FAILED]** Example heatmap visualization of attention weights showing how tokens attend to each other in a sequence.
>
> **Alt:** Heatmap visualization of attention weights over input tokens
>
> **Prompt:** A heatmap visualization of attention weights from a self-attention layer, showing token-to-token attention scores over a short input sequence, with tokens labeled on axes and color intensity indicating attention strength.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://aistudio.google.com/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image, description: Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.\nPlease retry in 1h57m54.118371771s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://aistudio.google.com/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'description': 'Generate content requests per day limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content requests per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'description': 'Generate content input tokens per minute limit exceeded on Free Tier. See https://aistudio.google.com/docs/rate-limits.', 'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '7074s'}]}}


By integrating these debugging and observability techniques, developers can gain actionable insights into self-attention models, leading to more reliable and interpretable transformer architectures.