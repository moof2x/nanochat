"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.

Addapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.

This file also contains an Aurora variant. Aurora ("A Leverage-Aware Optimizer for
Rectangular Matrices", Tilde Research, 2026; https://blog.tilderesearch.com/blog/aurora,
reference impl: https://github.com/tilde-research/aurora-release) replaces Muon's
plain polar factor with a damped iteration that alternates polar(.) and a row-norm
rebalancing step on tall (or transposed-tall) matrices, so that the orthogonalized
update sits on the joint Stiefel + row-oblique manifold. For square matrices Aurora
reduces to Muon. The key behavioural difference vs Muon+NorMuon is that Aurora
preserves polar precision while still producing row-uniform updates, which the
Aurora authors attribute to fewer "dead" MLP neurons.

In nanochat's optimizer the existing Muon path already includes:
    - Polar Express orthogonalization
    - NorMuon-style per-row variance reduction
    - Cautious-update masking
The Aurora kind below replaces the polar+variance-reduction pair with Aurora's
leverage-uniform polar (using the same Polar Express coefficients as the inner
orthogonalizer, for a fair head-to-head with the Muon path), and keeps the
cautious update + decoupled weight decay unchanged.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Weight decay (decoupled, applied before the update)
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# Canonical Polar Express coefficients from Amsel et al. 2025 (Algorithm 1 in
# https://arxiv.org/pdf/2505.16932, also at https://github.com/NoahAmsel/PolarExpress).
# These are the "as-published" coefficients with the standard 1.01 safety
# factor applied to all but the last iterate. Extending PE beyond 5 steps was
# the bottleneck for testing PE-8 in nanochat; the canonical table is needed
# because the safety_factor=2e-2/cushion=2 polynomials above were only fit
# for num_iters=5. We use this list when the caller asks for ns_steps > 5,
# or when --ns-coeffs=canonical is set.
#
# Note: the asymptotic coefficient is (1.875, -1.25, 0.375) — i.e. the
# degree-5 polynomial p(x) = 15x/8 - 10x^3/8 + 3x^5/8 with x rescaled, which
# has a super-attracting fixed point at x=1.
_polar_express_canonical_raw = [
    (8.28721201814563,  -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946,-2.908902115962949,  0.5518191394370137),
    (3.3184196573706015,-2.488488024314874,  0.51004894012372  ),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479,-1.2500016453999487, 0.3750001645474248),
    (1.875,             -1.25,                0.375              ),
]
_safety = 1.01
polar_express_coeffs_canonical = [
    (a / _safety, b / _safety**3, c / _safety**5)
    for (a, b, c) in _polar_express_canonical_raw[:-1]
] + [_polar_express_canonical_raw[-1]]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Variant of muon_step_fused that reads coefficients from
# polar_express_coeffs_canonical (the as-published Polar Express table from
# Amsel et al. 2025). Lets us run with up to 8 PE steps, or sweep ns_steps
# on the canonical schedule. Behaviour identical to muon_step_fused otherwise.
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused_canonical(
    stacked_grads: Tensor, stacked_params: Tensor,
    momentum_buffer: Tensor, second_momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, wd_t: Tensor, beta2_t: Tensor,
    ns_steps: int, red_dim: int,
) -> None:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs_canonical[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs_canonical[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
"""
Aurora step: Muon update with a leverage-uniform polar factor.

Reference: https://blog.tilderesearch.com/blog/aurora
            https://github.com/tilde-research/aurora-release

For a (tall) update X = G/||G||_F, define D_0 = I and iterate
    r_k       = row-norms of X_k
    D_k       = D_{k-1}^beta * diag(r_k)^(1-beta)
    X_{k+1}   = polar( sqrt(n/m) * D_k^{-1} X_k )
which alternately enforces row-norm uniformity and orthogonality. The final
iterate is on the Stiefel manifold; the diagonal preconditioner D suppresses
rows with large leverage. For wide matrices we transpose to tall first; for
square matrices we fall through to a single polar factor (Aurora == Muon).

We absorb the diagonal D into X explicitly (X <- D^{-1} X) so the inner loop
only touches the matmul-heavy part. The Aurora authors call this the "vanilla"
(non-Riemannian) Aurora: a damped fixed-point projection.

The variance-reduction step from NorMuon is omitted in this path: by Claim 1
of the Aurora paper, NorMuon's row normalization is in tension with the polar
constraint, and Aurora is designed to make that step unnecessary. We do keep
nanochat's cautious-update masking + decoupled weight decay (orthogonal to the
Aurora algorithmic novelty).
"""

# Same Polar Express coefficients used by Muon path: keeps the inner
# orthogonalizer comparable head-to-head with the existing Muon path.
@torch.compile(dynamic=False, fullgraph=True)
def aurora_step_fused(
    stacked_grads: Tensor,           # (K, m, n) - stacked gradients
    stacked_params: Tensor,          # (K, m, n) - stacked parameters
    momentum_buffer: Tensor,         # (K, m, n) - first moment buffer
    momentum_t: Tensor,              # () - 0-D CPU tensor, momentum coefficient
    lr_t: Tensor,                    # () - 0-D CPU tensor, learning rate
    wd_t: Tensor,                    # () - 0-D CPU tensor, weight decay
    ns_steps: int,                   # number of polar Newton-Schulz iterations
    pp_iterations: int,              # number of Aurora outer (row-rebalance) iterations
    pp_beta: float,                  # damping exponent for D update, in (0, 1]
    eps: float,                      # numerical floor for row norms
    do_aurora: bool,                 # True => tall(/wide-transposed); False => square (skip rebalance)
) -> None:
    """
    Fused Aurora step: nesterov_momentum -> aurora_polar -> cautious_update.
    All shape-static so torch.compile fully traces it.

    Signature mirrors muon_step_fused for shape compatibility with the
    DistMuonAdamW comm layout. red_dim and second_momentum_buffer are
    deliberately absent: Aurora replaces NorMuon's variance reduction.
    """
    # Nesterov momentum (identical to Muon path).
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Cast to bf16 (matching Muon path) for tensor-core friendly matmuls in polar.
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    m_dim = X.size(-2)
    n_dim = X.size(-1)
    transposed = m_dim < n_dim
    if transposed:
        # Make the "long" dimension the row axis so the leverage-anisotropy
        # statement (rows out-number columns) holds; transpose back at the end.
        X = X.mT

    # After (possible) transpose we have X of shape (..., M, N) with M >= N.
    M = X.size(-2)
    N = X.size(-1)
    target_row_sq = float(N) / float(M)  # uniform leverage value n/m

    if do_aurora:
        # Aurora damped iteration. We track D implicitly by folding it into X.
        for k in range(pp_iterations):
            # Polar(X) using the same Polar Express coefficients as Muon path.
            # Re-normalise spectral norm at the start of *every* inner polar so
            # the Newton-Schulz iteration stays in its convergence basin even
            # after the row-rescaling step (which can change the spectrum).
            P = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
            # Inner Polar Express loop (ns_steps coefficients), tall branch.
            for a, b, c in polar_express_coeffs[:ns_steps]:
                A = P.mT @ P
                B = b * A + c * (A @ A)
                P = a * P + P @ B
            if k < pp_iterations - 1:
                # Update X = D^{-1} P with D from row norms of P, so the next
                # polar call sees a row-mass-balanced matrix. We rescale to
                # match the target row-squared-norm n/m, raised to the damping
                # exponent pp_beta to suppress oscillations between iterates.
                row_sq = P.float().pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                scale = (target_row_sq / row_sq).pow(pp_beta * 0.5).to(P.dtype)
                X = P * scale
            else:
                X = P
    else:
        # Square: vanilla polar (Aurora collapses to Muon here).
        P = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = P.mT @ P
            B = b * A + c * (A @ A)
            P = a * P + P @ B
        X = P

    if transposed:
        X = X.mT
    g = X

    # Spectral aspect-ratio scaling: this is folded into lr_t by the caller
    # (see _step_aurora). Cautious weight decay + parameter update.
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# -----------------------------------------------------------------------------
# Variant of aurora_step_fused that uses the canonical Polar Express
# coefficient table (Amsel et al. 2025) for the inner orthogonalizer. Lets us
# run Aurora with up to 8 PE steps for a higher-precision polar.
@torch.compile(dynamic=False, fullgraph=True)
def aurora_step_fused_canonical(
    stacked_grads: Tensor, stacked_params: Tensor, momentum_buffer: Tensor,
    momentum_t: Tensor, lr_t: Tensor, wd_t: Tensor,
    ns_steps: int, pp_iterations: int, pp_beta: float, eps: float, do_aurora: bool,
) -> None:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g
    m_dim = X.size(-2)
    n_dim = X.size(-1)
    transposed = m_dim < n_dim
    if transposed:
        X = X.mT
    M = X.size(-2)
    N = X.size(-1)
    target_row_sq = float(N) / float(M)
    if do_aurora:
        for k in range(pp_iterations):
            P = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
            for a, b, c in polar_express_coeffs_canonical[:ns_steps]:
                A = P.mT @ P
                B = b * A + c * (A @ A)
                P = a * P + P @ B
            if k < pp_iterations - 1:
                row_sq = P.float().pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                scale = (target_row_sq / row_sq).pow(pp_beta * 0.5).to(P.dtype)
                X = P * scale
            else:
                X = P
    else:
        P = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
        for a, b, c in polar_express_coeffs_canonical[:ns_steps]:
            A = P.mT @ P
            B = b * A + c * (A @ A)
            P = a * P + P @ B
        X = P
    if transposed:
        X = X.mT
    g = X
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW tensors
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Aurora tensors (separate to avoid recompilation collisions with Muon path).
        self._aurora_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aurora_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aurora_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Dispatch to the right fused kernel based on which Polar Express
        # coefficient table the caller asked for. Default ('nanochat') keeps
        # the historical 5-step coefficients. 'canonical' uses the Amsel et al.
        # 2025 table (supports up to 8 PE steps).
        coeffs = group.get("coeffs", "nanochat")
        kernel = muon_step_fused if coeffs == "nanochat" else muon_step_fused_canonical
        kernel(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _step_aurora(self, group: dict) -> None:
        """
        Aurora update for all params in the group (stacked for efficiency).
        Same buffer-stacking strategy as _step_muon. No second-moment buffer:
        Aurora replaces NorMuon's variance reduction with the row-rebalancing
        polar iteration.
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        self._aurora_momentum_t.fill_(group["momentum"])
        # Match Muon's spectral aspect-ratio scaling on the LR (Muon convention).
        self._aurora_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._aurora_wd_t.fill_(group["weight_decay"])

        # Aurora is meaningful only on rectangular matrices (M != N). For square
        # matrices the row-norm rebalancing has no degrees of freedom (claim 1
        # in the Aurora paper) and Aurora reduces to Muon's polar update.
        do_aurora = (shape[-2] != shape[-1])

        coeffs = group.get("coeffs", "nanochat")
        kernel = aurora_step_fused if coeffs == "nanochat" else aurora_step_fused_canonical
        kernel(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            self._aurora_momentum_t,
            self._aurora_lr_t,
            self._aurora_wd_t,
            group["ns_steps"],
            group["pp_iterations"],
            float(group["pp_beta"]),
            float(group.get("aurora_eps", 1e-7)),
            bool(do_aurora),
        )

        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            elif group['kind'] == 'aurora':
                self._step_aurora(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aurora_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aurora_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._aurora_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            coeffs = group.get("coeffs", "nanochat")
            kernel = muon_step_fused if coeffs == "nanochat" else muon_step_fused_canonical
            kernel(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _reduce_aurora(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for an Aurora group. Identical comm pattern
        to Muon: stack grads, zero-pad to a multiple of world_size, reduce_scatter."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_aurora(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Aurora updates on this rank's chunk, launch all_gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)

        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            self._aurora_momentum_t.fill_(group["momentum"])
            self._aurora_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._aurora_wd_t.fill_(group["weight_decay"])

            do_aurora = (shape[-2] != shape[-1])

            coeffs = group.get("coeffs", "nanochat")
            kernel = aurora_step_fused if coeffs == "nanochat" else aurora_step_fused_canonical
            kernel(
                grad_chunk[:num_owned],
                stacked_owned,
                state["momentum_buffer"][:num_owned],
                self._aurora_momentum_t,
                self._aurora_lr_t,
                self._aurora_wd_t,
                group["ns_steps"],
                group["pp_iterations"],
                float(group["pp_beta"]),
                float(group.get("aurora_eps", 1e-7)),
                bool(do_aurora),
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            elif group['kind'] == 'aurora':
                reduce_infos.append(self._reduce_aurora(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            elif group['kind'] == 'aurora':
                self._compute_aurora(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)
