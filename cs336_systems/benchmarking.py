import argparse
from cs336_basics.model import BasicsTransformerLM
import torch
import timeit

# Model configs from Table 1
MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large", "xl", "2.7B"],
        default="small",
        help="Model size configuration to use",
    )
    parser.add_argument("--context-length", type=int, default=512, help="Context length for the model")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for benchmarking")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup steps")
    parser.add_argument("--steps", type=int, default=10, help="Number of benchmark steps")
    parser.add_argument("--no-backward", action="store_true", help="Exclude backward pass from benchmarking")
    return parser.parse_args()

def init_model(d_model, d_ff, n_heads, n_layers, vocab_size, context_length, rope_theta):
    model = BasicsTransformerLM(
        d_model=d_model, 
        d_ff=d_ff, 
        num_heads=n_heads, 
        num_layers=n_layers,
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta
    )
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
        if torch.cuda.is_available():
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = timeit.default_timer()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time

if __name__ == "__main__":
    args = parse_args()
    config = MODEL_CONFIGS[args.model_size]

    vocab_size = 10000
    rope_theta = 100000.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Benchmarking model-size={args.model_size}, batch-size={args.batch_size}, context-length={args.context_length}")

    model = init_model(
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        n_heads=config["num_heads"],
        n_layers=config["num_layers"],
        vocab_size=vocab_size,
        context_length=args.context_length,
        rope_theta=rope_theta
    ).to(device)

    avg_time_forward = benchmark(
        model, args.batch_size, args.context_length, vocab_size, 
        args.warmup, args.steps, include_backward=False
    )
    print(f"Average forward pass time: {avg_time_forward:.4f} seconds")

    if not args.no_backward:
        avg_time_forward_backward = benchmark(
            model, args.batch_size, args.context_length, vocab_size, 
            args.warmup, args.steps, include_backward=True
        )
        print(f"Average forward + backward pass time: {avg_time_forward_backward:.4f} seconds")
        print(f"Estimated average backward pass time: {avg_time_forward_backward - avg_time_forward:.4f} seconds")