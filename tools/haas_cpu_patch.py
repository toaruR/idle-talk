"""Haas-ja-v1.0 を GPU なしで動かすための差し替え (比較・検証用)。

Haas-ja の予測ネットワーク (KDA, 3.9M パラメータ) は flash-linear-attention の Triton カーネルで動き、
公式には NVIDIA GPU が要る。CPU では TRITON_INTERPRET=1 で同じカーネルを解釈実行できるが、1 発話に数十秒〜数分かかる。
そこで、デコードの 1 ステップごとに呼ばれる 3 つのカーネルを、同じ式の PyTorch 実装に置き換える:

- fused_recurrent_kda (KDA の再帰更新。Haas の設定 = lower_bound なし・負の固有値なし・state_v_first)
- ShortConvolution.step (短い畳み込みの 1 ステップ。キャッシュをその場で更新)
- FusedRMSNormGated.forward (ゲート付き RMSNorm)

それ以外 (q_len > 64 の chunk モードなど、デコードでは通らない経路) は Triton インタプリタのまま動く。
apply() は torch / fla / haas_asr を import する前に呼ぶ (TRITON_INTERPRET を先に設定するため)。
tools/asr_bench_ja.py --device cpu が使う。置き換えが元のカーネルと同じ値を返すことは verify() で確かめられる。
"""
from __future__ import annotations

import os


def _fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
                         output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                         use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False, lower_bound=None,
                         state_v_first=False, cu_seqlens=None, **kwargs):
    import torch
    import torch.nn.functional as F

    if cu_seqlens is not None or lower_bound is not None or kwargs.get("ssm_state_indices") is not None:
        raise NotImplementedError("haas_cpu_patch: unsupported fused_recurrent_kda option")
    dtype = v.dtype
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)
    if use_gate_in_kernel:
        if dt_bias is not None:
            g = g + dt_bias.float().view(HV, K)
        g = -A_log.float().exp().view(HV, 1) * F.softplus(g)
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid() * (2 if allow_neg_eigval else 1)
    # S: [B, HV, K, V] (state_v_first なら保存形は [B, HV, V, K])
    if initial_state is None:
        S = q.new_zeros(B, HV, K, V)
    else:
        S = initial_state.float()
        S = S.transpose(-1, -2) if state_v_first else S
        S = S.clone()
    o = q.new_zeros(B, T, HV, V)
    for t in range(T):
        S = S * g[:, t].exp()[..., None]
        k_t = k[:, t]
        v_t = (v[:, t] - (S * k_t[..., None]).sum(-2)) * beta[:, t][..., None]
        S = S + k_t[..., None] * v_t[..., None, :]
        o[:, t] = (S * q[:, t][..., None]).sum(-2)
    final = None
    if output_final_state:  # 公開の fused_recurrent_kda は inplace_final_state=False: h0 は書き換えず新しく返す
        final = (S.transpose(-1, -2) if state_v_first else S).contiguous()
    return o.to(dtype), final


def _short_conv_step(self, x, residual, cache, output_final_state=False, cu_seqlens=None):
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    if cu_seqlens is not None:
        raise NotImplementedError("haas_cpu_patch: cu_seqlens")
    B, _, D = x.shape
    W = self.kernel_size[0]
    if cache is None:
        cache = x.new_zeros(B, D, W)
    cache.copy_(torch.cat([cache[..., 1:], x[:, 0, :, None].to(cache.dtype)], dim=-1))
    y = (cache.float() * rearrange(self.weight, "d 1 w -> d w").float()).sum(-1)
    if self.bias is not None:
        y = y + self.bias.float()
    if self.activation in ("silu", "swish"):
        y = F.silu(y)
    y = y.to(x.dtype).unsqueeze(1)
    if residual is not None:
        y = y + residual
    return y, (cache if output_final_state else None)


def _rms_norm_gated(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
    import torch

    if residual is not None or prenorm:
        raise NotImplementedError("haas_cpu_patch: residual/prenorm")
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
    if self.weight is not None:
        y = y * self.weight.float()
    if self.bias is not None:
        y = y + self.bias.float()
    gf = g.float()
    y = y * (gf.sigmoid() if self.activation == "sigmoid" else gf * gf.sigmoid())
    return y.to(x.dtype)


_ORIGINAL = {}


def apply() -> None:
    """CPU 用に差し替える。何度呼んでもよい。"""
    os.environ.setdefault("TRITON_INTERPRET", "1")
    from triton.runtime import autotuner

    import fla.layers.kda as kda_layer
    from fla.modules.conv.short_conv import ShortConvolution
    from fla.modules.fused_norm_gate import FusedRMSNormGated

    if _ORIGINAL:
        return
    _ORIGINAL.update(fused_recurrent_kda=kda_layer.fused_recurrent_kda, step=ShortConvolution.step,
                     norm=FusedRMSNormGated.forward, bench=autotuner.Autotuner._bench)
    # インタプリタではカーネルの時間を測れない。設定の違いはブロックの大きさだけで結果は同じなので、測らずに選ぶ。
    autotuner.Autotuner._bench = lambda self, *a, config, **k: [0.0, 0.0, 0.0]
    kda_layer.fused_recurrent_kda = _fused_recurrent_kda
    ShortConvolution.step = _short_conv_step
    FusedRMSNormGated.forward = _rms_norm_gated


def verify(seed: int = 0) -> None:
    """置き換えた 3 つが、元の Triton カーネル (インタプリタ) と同じ値を返すか確かめる。"""
    import torch

    apply()
    from fla.modules.conv.short_conv import ShortConvolution
    from fla.modules.fused_norm_gate import FusedRMSNormGated

    torch.manual_seed(seed)
    B, H, K, V = 3, 5, 64, 64
    for T in (1, 3):
        q, k = torch.randn(B, T, H, K), torch.randn(B, T, H, K)
        v, g, beta = torch.randn(B, T, H, V), torch.randn(B, T, H, K), torch.randn(B, T, H)
        A_log, dt_bias = torch.randn(H) * 0.5, torch.randn(H * K) * 0.5
        h0 = torch.randn(B, H, V, K) * 0.1
        kw = dict(A_log=A_log, dt_bias=dt_bias, output_final_state=True, use_qk_l2norm_in_kernel=True,
                  use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, state_v_first=True)
        ref_h0, new_h0 = h0.clone(), h0.clone()
        o_ref, s_ref = _ORIGINAL["fused_recurrent_kda"](q, k, v, g, beta, initial_state=ref_h0, **kw)
        o_new, s_new = _fused_recurrent_kda(q, k, v, g, beta, initial_state=new_h0, **kw)
        torch.testing.assert_close(o_new, o_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(s_new, s_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(new_h0, ref_h0, rtol=1e-4, atol=1e-5)

    conv = ShortConvolution(hidden_size=320, kernel_size=4, bias=False, activation="silu")
    x, cache = torch.randn(B, 1, 320), torch.randn(B, 320, 4)
    c_ref, c_new = cache.clone(), cache.clone()
    y_ref, _ = _ORIGINAL["step"](conv, x, None, c_ref, True)
    y_new, _ = _short_conv_step(conv, x, None, c_new, True)
    torch.testing.assert_close(y_new, y_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(c_new, c_ref)

    norm = FusedRMSNormGated(64, activation="sigmoid", eps=1e-5)
    with torch.no_grad():
        norm.weight.normal_()
    x, gate = torch.randn(B, 2, H, 64), torch.randn(B, 2, H, 64)
    torch.testing.assert_close(_rms_norm_gated(norm, x, gate), _ORIGINAL["norm"](norm, x, gate),
                               rtol=1e-4, atol=1e-5)
    print("haas_cpu_patch: fused_recurrent_kda / short conv step / gated RMSNorm match the Triton kernels")


if __name__ == "__main__":
    verify()
