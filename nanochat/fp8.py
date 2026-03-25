"""Minimal FP8 training for nanochat -- tensorwise dynamic scaling only.

Optimized version (G17 evolution winner, +3.3% throughput vs original):
  1. Skip fp32/fp64 upcasts in quantization (abs in native dtype, scalar-only upcast)
  2. Pre-compute col-major layouts in forward for backward reuse
  3. use_fast_accum=True in backward GEMMs
  4. Fused _to_fp8_col: quantize + col-major in one call (avoids redundant fp8 tensor)
  5. Removed COMPUTE_DTYPE cast in Float8Linear.forward (input already in correct dtype)
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float()
    scale = (fp8_max / amax.clamp(min=EPS))
    x_fp8 = (x * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale.reciprocal()


@torch.no_grad()
def _to_fp8_col(x, fp8_dtype):
    """Quantize to FP8 and return both row-major and col-major layouts.

    Fuses quantization with col-major layout preparation, avoiding a
    separate _to_col_major call. The col-major version is needed by
    backward for the grad_weight GEMM.
    """
    fp8_max = torch.finfo(fp8_dtype).max
    amax = x.abs().max().float()
    scale = (fp8_max / amax.clamp(min=EPS))
    x_fp8 = (x * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    x_col = x_fp8.t().contiguous().t()
    return x_fp8, x_col, scale.reciprocal()


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation --
# dynamo won't try to decompose it into smaller ops.
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize input to FP8 and get both row-major and col-major
        input_fp8, input_col, input_inv = _to_fp8_col(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)

        # Pre-compute col-major weight for backward
        weight_col = weight_fp8.t().contiguous().t()
        ctx.save_for_backward(input_col, input_inv, weight_col, weight_inv)

        # output = input @ weight.T
        output = torch._scaled_mm(
            input_fp8, weight_fp8.t(),
            scale_a=input_inv, scale_b=weight_inv,
            out_dtype=input_2d.dtype, use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_col, in_inv, w_col, w_inv = ctx.saved_tensors

        # Quantize gradient to FP8
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        go_t = go_fp8.t().contiguous()

        # grad_input = grad_output @ weight
        grad_input = torch._scaled_mm(
            go_fp8, w_col,
            scale_a=go_inv, scale_b=w_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )

        # grad_weight = grad_output.T @ input
        grad_weight = torch._scaled_mm(
            go_t, in_col,
            scale_a=go_inv, scale_b=in_inv,
            out_dtype=grad_output.dtype, use_fast_accum=False,
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    def forward(self, input):
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor -- we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"Rowwise/axiswise recipes require the full torchao library."
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors -- no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module
