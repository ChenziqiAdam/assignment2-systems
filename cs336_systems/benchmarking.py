import cs336_basics
import torch
import torch
import timeit

def init_model(d_model, d_ff, n_heads, n_layers):
    model = cs336_basics.model.BasicsTransformerLM(d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_layers=n_layers)
    return model

def generate_batch(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids

def benchmark(model, batch_size, seq_len, vocab_size, warmup_steps, benchmark_steps, include_backward=False):
    
    device = next(model.parameters()).device
    input_ids = generate_batch(batch_size, seq_len, vocab_size).to(device)
    
    # Warmup
    for _ in range(warmup_steps):
        logits = model(input_ids)
        if include_backward:
            loss = logits.mean()
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_steps):
        start = timeit.default_timer()
        logits = model(input_ids)
        if include_backward:
            loss = logits.mean()
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    return avg_time

if __name__ == "__main__":
    d_model = 512
    d_ff = 2048
    n_heads = 8
    n_layers = 6
    batch_size = 32
    seq_len = 128
    vocab_size = 10000
    warmup_steps = 5
    benchmark_steps = 10
    
    model = init_model(d_model, d_ff, n_heads, n_layers).cuda()
    
    avg_time_forward = benchmark(model, batch_size, seq_len, vocab_size, warmup_steps, benchmark_steps, include_backward=False)
    print(f"Average forward pass time: {avg_time_forward:.4f} seconds")
    
    avg_time_forward_backward = benchmark(model, batch_size, seq_len, vocab_size, warmup_steps, benchmark_steps, include_backward=True)
    print(f"Average backward pass time: {avg_time_forward_backward - avg_time_forward:.4f} seconds")