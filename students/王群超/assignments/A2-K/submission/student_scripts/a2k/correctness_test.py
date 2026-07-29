import torch
import math
import json
import os

def reference_attn_and_lse(Q, K, V, is_causal=False):
    """测试基准"""
    
    n_query = Q.shape[1]
    n_key = K.shape[1]
    scale = 1.0 / math.sqrt(Q.shape[-1])
    
    S = Q @ K.transpose(-2, -1) * scale

    if is_causal :
        col_idx = torch.arange(n_query, device=Q.device).unsqueeze(0)
        row_idx = torch.arange(n_key, device=K.device).unsqueeze(1)
        mask = col_idx > row_idx
        S = S.masked_fill(mask, float('-inf'))

    m = S.max(dim=-1, keepdim=True).values
    lse = m.squeeze(-1) + torch.log(torch.exp(S - m).sum(dim=-1))

    P = torch.exp(S - lse.unsqueeze(-1))
    O = P @ V

    return O, lse

def correctness_test(seed, head_dim, is_causal, dtype, impl, seq_len=128, batch_size=4):
    torch.manual_seed(seed)
    device = "cuda"

    status = 'success'

    try:
        Q = torch.randn((batch_size, seq_len, head_dim), device=device, dtype=dtype, requires_grad=True)
        K = torch.randn((batch_size, seq_len, head_dim), device=device, dtype=dtype, requires_grad=True)
        V = torch.randn((batch_size, seq_len, head_dim), device=device, dtype=dtype, requires_grad=True)
        dO = torch.randn((batch_size, seq_len, head_dim), device=device, dtype=dtype, requires_grad=True)
        
        
        #实现方式选择
        if impl == 'pytorch':
            from tests.adapters import get_flashattention_autograd_function_pytorch
            cls = get_flashattention_autograd_function_pytorch()
        else:
            from tests.adapters import get_flashattention_autograd_function_triton
            cls = get_flashattention_autograd_function_triton()

        #参考数据
        Q_ref = Q.float().detach().requires_grad_(True)
        K_ref = K.float().detach().requires_grad_(True)
        V_ref = V.float().detach().requires_grad_(True)
        dO_ref = dO.float()
        O_ref, L_ref = reference_attn_and_lse(Q_ref, K_ref, V_ref, is_causal)

        # Forward
        O_student = cls.apply(Q, K, V, is_causal)

        # 验证 O
        max_abs_err_O = (O_student.float() - O_ref).abs().max().item()
        max_rel_err_O = ((O_student.float() - O_ref).abs() / (O_ref.abs() + 1e-8)).max().item()
        
        # 验证 L
        L_student = O_student.grad_fn.saved_tensors[-1]  # 最后一个 saved tensor
        max_abs_err_L = (L_student.float() - L_ref).abs().max().item()

        O_student.backward(dO)
        dQ_student = Q.grad
        dK_student = K.grad
        dV_student = V.grad

        O_ref.backward(dO_ref)
        dQ_ref = Q_ref.grad
        dK_ref = K_ref.grad
        dV_ref = V_ref.grad

        max_abs_err_dQ = (dQ_student.float() - dQ_ref).abs().max().item()
        max_abs_err_dK = (dK_student.float() - dK_ref).abs().max().item()
        max_abs_err_dV = (dV_student.float() - dV_ref).abs().max().item()

        is_pass = max_abs_err_O < 1e-2 and max_abs_err_L < 1e-2 and max_abs_err_dQ < 1e-2 and max_abs_err_dK < 1e-2 and max_abs_err_dV < 1e-2
    
    except torch.cuda.OutOfMemoryError:
        max_abs_err_O = max_rel_err_O = max_abs_err_L = max_abs_err_dQ = max_abs_err_dK = max_abs_err_dV = is_pass ='None'
        status = 'oom'
        torch.cuda.empty_cache()
                
    
    except RuntimeError as e:
        max_abs_err_O = max_rel_err_O = max_abs_err_L = max_abs_err_dQ = max_abs_err_dK = max_abs_err_dV = is_pass = None
        print(f"RuntimeError: {e}")
        import traceback
        traceback.print_exc()
        status = 'error'
        torch.cuda.empty_cache()
                
        
    except Exception as e :
        max_abs_err_O = max_rel_err_O = max_abs_err_L = max_abs_err_dQ = max_abs_err_dK = max_abs_err_dV = is_pass ='None'
        print(f"Exception:{e}")
        status= 'error'
        torch.cuda.empty_cache()

    return {
        'seed': seed,
        'head_dim': head_dim,
        'is_causal': is_causal,
        'impl': impl,
        'dtype': str(dtype),
        'max_abs_err_O': max_abs_err_O,
        'max_rel_err_O': max_rel_err_O,
        'max_abs_err_L': max_abs_err_L,
        'max_abs_err_dQ': max_abs_err_dQ,
        'max_abs_err_dK': max_abs_err_dK,
        'max_abs_err_dV': max_abs_err_dV,
        'pass': is_pass,
        'status':status
    }


results = []
for seed in [42, 123, 456]:
    for head_dim in [32, 64, 128]:
        for is_causal in [False, True]:
            for impl in ['pytorch', 'triton']:
                for dtype in [torch.float32, torch.bfloat16]:
                    result = correctness_test(seed, head_dim, is_causal, dtype, impl)
                    results.append(result)
                    print(f"{impl} seed={seed} head_dim={head_dim} causal={is_causal} dtype={dtype}:{result['pass']}:{result['status']}")

file_path = 'local_results/correctness.json'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(file_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} results to {file_path}")

