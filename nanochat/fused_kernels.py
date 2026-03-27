"""
Fused Triton kernels for nanochat performance optimization.

1. Chunked cross-entropy with tanh softcap:
   - Avoids materializing full (B*T, vocab_size) logits tensor
   - Recomputes logits per chunk in backward (activation checkpointing on logits)
   - Huge memory savings + reduced memory bandwidth pressure

2. Fused MLP (relu_squared inside the matmul tile):
   - Fuses linear -> relu -> square into one kernel pass
   - Eliminates the 4x-expanded intermediate activation write+read
   - Port of modded-nanogpt's FusedLinearReLUSquareFunction using TMA descriptors
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# =============================================================================
# 1. Chunked Cross-Entropy with Tanh Softcap
# =============================================================================

class ChunkedSoftcapCrossEntropy(torch.autograd.Function):
    """
    Memory-efficient cross-entropy that never materializes the full logits tensor.

    Forward:
      For each chunk of the sequence:
        logits_chunk = x_chunk @ weight.T   (BF16 matmul)
        logits_chunk = logits_chunk[:, :vocab_size].float()
        capped = softcap * tanh(logits_chunk / softcap)
        loss_chunk = cross_entropy(capped, targets_chunk)
      Accumulate loss across chunks.

    Backward:
      Recompute logits per chunk (activation checkpointing on logits).
      Compute grad_x and grad_weight per chunk, accumulate.

    This avoids storing the full (B*T, padded_vocab_size) logits tensor,
    which at B=8, T=2048, V=32768 would be ~2GB in float32.
    """

    @staticmethod
    def forward(ctx, x, weight, targets, softcap, vocab_size, num_chunks):
        """
        x: (B*T, D) in bf16
        weight: (padded_V, D) in fp32 (will be cast to bf16 for matmul)
        targets: (B*T,) int64
        softcap: float
        vocab_size: int (unpadded)
        num_chunks: int
        """
        B_T, D = x.shape
        chunk_size = (B_T + num_chunks - 1) // num_chunks

        total_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        n_valid = torch.tensor(0, device=x.device, dtype=torch.long)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, B_T)
            if start >= end:
                break

            x_chunk = x[start:end]
            t_chunk = targets[start:end]

            # Compute logits for this chunk
            logits = F.linear(x_chunk, weight.to(dtype=x.dtype))  # (chunk, padded_V)
            logits = logits[:, :vocab_size].float()  # slice padding, cast to fp32
            logits = softcap * torch.tanh(logits / softcap)  # tanh softcap

            # Cross-entropy for this chunk
            chunk_loss = F.cross_entropy(logits, t_chunk, ignore_index=-1, reduction='sum')
            total_loss = total_loss + chunk_loss

            # Count valid tokens (stay on GPU, no .item())
            n_valid = n_valid + (t_chunk != -1).sum()

        # Average over valid tokens
        total_loss = total_loss / n_valid.clamp(min=1).float()

        ctx.save_for_backward(x, weight, targets, n_valid)
        ctx.softcap = softcap
        ctx.vocab_size = vocab_size
        ctx.num_chunks = num_chunks

        return total_loss

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, targets, n_valid = ctx.saved_tensors
        softcap = ctx.softcap
        vocab_size = ctx.vocab_size
        num_chunks = ctx.num_chunks

        B_T, D = x.shape
        chunk_size = (B_T + num_chunks - 1) // num_chunks

        grad_x = torch.zeros_like(x)
        grad_weight = torch.zeros_like(weight)

        scale = grad_output / n_valid.clamp(min=1).float()

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, B_T)
            if start >= end:
                break

            x_chunk = x[start:end]
            t_chunk = targets[start:end]

            # Recompute logits (activation checkpointing)
            logits_raw = F.linear(x_chunk, weight.to(dtype=x.dtype))  # (chunk, padded_V)
            logits_raw = logits_raw[:, :vocab_size].float()

            # tanh softcap forward
            z = logits_raw / softcap
            tanh_z = torch.tanh(z)
            capped = softcap * tanh_z  # (chunk, V)

            # Softmax probabilities
            probs = F.softmax(capped, dim=-1)  # (chunk, V)

            # Gradient of cross-entropy w.r.t. capped logits
            # d_loss/d_capped = (probs - one_hot) / n_valid * grad_output
            one_hot = torch.zeros_like(probs)
            valid_mask = t_chunk != -1
            if valid_mask.any():
                one_hot[valid_mask, t_chunk[valid_mask]] = 1.0
            grad_capped = (probs - one_hot) * scale  # (chunk, V)

            # Gradient through tanh softcap: d_capped/d_raw = (1 - tanh^2(z))
            grad_raw = grad_capped * (1 - tanh_z * tanh_z)  # (chunk, V)

            # Need to pad grad_raw back to padded_vocab_size for weight grad
            padded_V = weight.shape[0]
            if vocab_size < padded_V:
                grad_raw_padded = torch.zeros(end - start, padded_V, device=grad_raw.device, dtype=grad_raw.dtype)
                grad_raw_padded[:, :vocab_size] = grad_raw
            else:
                grad_raw_padded = grad_raw

            # grad_x_chunk = grad_raw_padded @ weight (bf16 matmul)
            grad_raw_bf16 = grad_raw_padded.to(x.dtype)
            grad_x[start:end] = F.linear(grad_raw_bf16, weight.to(dtype=x.dtype).t()).to(x.dtype)

            # grad_weight += grad_raw_padded.T @ x_chunk
            grad_weight += grad_raw_padded.t().to(weight.dtype) @ x_chunk.float()

        return grad_x, grad_weight, None, None, None, None


def chunked_softcap_cross_entropy(x, weight, targets, softcap=15.0, vocab_size=None, num_chunks=4):
    """
    Drop-in replacement for the logits + softcap + cross_entropy sequence.

    Args:
        x: (B*T, D) hidden states (bf16)
        weight: (padded_V, D) lm_head weight
        targets: (B*T,) target token ids
        softcap: softcap value (default 15)
        vocab_size: unpadded vocab size
        num_chunks: number of chunks (more chunks = less memory, slightly more compute)
    """
    if vocab_size is None:
        vocab_size = weight.shape[0]
    return ChunkedSoftcapCrossEntropy.apply(x, weight, targets, softcap, vocab_size, num_chunks)


# =============================================================================
# 2. Fused MLP: relu(x @ W1.T)^2 @ W2.T
# =============================================================================
# Port of modded-nanogpt's FusedLinearReLUSquareFunction using TMA descriptors.
# Fuses: linear1 -> relu -> square -> linear2 in forward
# And the corresponding backward pass.

@triton.jit
def _fused_mlp_fwd_kernel(
    a_desc, b_desc, c_desc, aux_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Forward: computes relu(x @ W1.T)^2, storing both pre-activation and post-activation."""
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        # Split into two halves for better TMA store throughput
        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        # First half: pre-activation -> relu^2
        c0 = acc0.to(dtype)
        c_desc.store([offs_am_c, offs_bn_c], c0)
        c0_post = tl.maximum(c0, 0)
        c0_post = c0_post * c0_post
        aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        # Second half
        c1 = acc1.to(dtype)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
        c1_post = tl.maximum(c1, 0)
        c1_post = c1_post * c1_post
        aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


@triton.jit
def _fused_mlp_bwd_kernel(
    a_desc, b_desc, c_desc, aux_desc,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Backward: computes d_pre = 2 * relu(pre) * (grad @ W2) element-wise fused with matmul."""
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        # First half: multiply by d(relu^2)/d(pre) = 2*relu(pre)
        c0 = acc0.to(dtype)
        c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
        c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)
        c_desc.store([offs_am_c, offs_bn_c], c0)

        # Second half
        c1 = acc1.to(dtype)
        c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
        c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)
        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)


def _fused_linear_relu_square(a, b, aux=None):
    """
    Low-level fused matmul + relu^2.
    If aux is None: forward pass, computes relu(a @ b.T)^2, returns (pre_activation, post_activation)
    If aux is provided: backward pass, computes 2*relu(aux) * (a @ b.T), returns d_pre
    """
    M, K = a.shape
    N, K2 = b.shape
    assert K == K2
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    FORWARD = aux is None
    if FORWARD:
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    kernel = _fused_mlp_fwd_kernel if FORWARD else _fused_mlp_bwd_kernel
    kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        num_stages=num_stages,
        num_warps=num_warps,
    )

    if FORWARD:
        return c, aux  # pre_activation, post_activation (relu^2)
    else:
        return c  # d_pre


class FusedLinearReLUSquareFunction(torch.autograd.Function):
    """
    Fused: relu(x @ W1.T)^2 @ W2.T

    Forward fuses linear1 + relu + square into one kernel.
    Backward reuses the pre-activation from forward to compute gradients.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        # x: (B, T, D) or (B*T, D)
        x_2d = x.view(-1, x.shape[-1])
        pre, post = _fused_linear_relu_square(x_2d, W1)  # pre = x @ W1.T, post = relu(pre)^2
        out = post @ W2  # (B*T, D_out)
        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_2d = x.view(-1, x.shape[-1]).contiguous()
        grad_2d = grad_output.view(-1, grad_output.shape[-1]).contiguous()

        # grad_W2 = post.T @ grad_output
        dW2 = post.T @ grad_2d

        # d_post = grad_output @ W2.T, then chain through relu^2:
        # d_pre = d_post * 2 * relu(pre) = fused_backward(grad_output, W2, pre)
        # W2 is (hdim, D), so we want d_post = grad @ W2.T -- but the fused kernel
        # computes a @ b.T, so we pass (grad_2d, W2) which computes grad_2d @ W2.T.
        # BUT W2 shape is (hdim, D), and grad_2d is (B*T, D).
        # grad_2d @ W2.T = (B*T, D) @ (D, hdim) = (B*T, hdim) -- correct!
        dpre = _fused_linear_relu_square(grad_2d, W2.contiguous(), aux=pre)

        # grad_W1 = dpre.T @ x
        dW1 = dpre.T @ x_2d

        # grad_x = dpre @ W1
        dx = dpre @ W1

        return dx.view(x.shape), dW1, dW2


# Convenience function
fused_mlp = FusedLinearReLUSquareFunction.apply
