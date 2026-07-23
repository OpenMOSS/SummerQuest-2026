import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float = 10000.0,
        max_seq_len: int | None = None,
        device=None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE")
        if theta <= 0:
            raise ValueError("theta must be positive")
        if max_seq_len is not None and max_seq_len < 0:
            raise ValueError("max_seq_len must be non-negative or None")

        self.theta = theta
        self.d_k = d_k
        cache_shape = (0, d_k // 2)
        self.register_buffer(
            "cos",
            torch.empty(cache_shape, device=device),
            persistent=False,
        )
        self.register_buffer(
            "sin",
            torch.empty(cache_shape, device=device),
            persistent=False,
        )

        if max_seq_len:
            self._grow_cache(max_seq_len)

    def _grow_cache(self, required_seq_len: int) -> None:
        cached_seq_len = self.cos.shape[0]
        if required_seq_len <= cached_seq_len:
            return

        # Geometric growth avoids rebuilding the cache on every generated token.
        new_cache_len = max(required_seq_len, max(1, cached_seq_len * 2))
        positions = torch.arange(
            cached_seq_len,
            new_cache_len,
            device=self.cos.device,
            dtype=torch.float32,
        )
        dims = torch.arange(
            0,
            self.d_k,
            2,
            device=self.cos.device,
            dtype=torch.float32,
        )
        inv_freq = 1.0 / (self.theta ** (dims / self.d_k))
        angles = torch.outer(positions, inv_freq)

        new_cos = torch.cos(angles).to(dtype=self.cos.dtype)
        new_sin = torch.sin(angles).to(dtype=self.sin.dtype)
        self.cos = torch.cat((self.cos, new_cos), dim=0)
        self.sin = torch.cat((self.sin, new_sin), dim=0)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected x.shape[-1] to be d_k={self.d_k}, got {x.shape[-1]}"
            )
        if token_positions.numel() == 0:
            return x
        if (
            token_positions.dtype == torch.bool
            or token_positions.is_floating_point()
            or token_positions.is_complex()
        ):
            raise TypeError("token_positions must contain integer positions")

        positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        if not torch.compiler.is_compiling():
            min_position = int(positions.min().item())
            if min_position < 0:
                raise ValueError("token_positions must be non-negative")

            required_seq_len = int(positions.max().item()) + 1
            self._grow_cache(required_seq_len)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos[positions].to(device=x.device, dtype=x.dtype)
        sin = self.sin[positions].to(device=x.device, dtype=x.dtype)

        # Multi-head attention adds a head axis that token_positions does not have.
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        res = torch.empty_like(x)

        res[..., 0::2] = x_even * cos - x_odd * sin
        res[..., 1::2] = x_even * sin + x_odd * cos
        return res