import torch
from gpt import GPTConfig, GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('model.pt', map_location=device, weights_only=False)
cfg = checkpoint['config']
stoi = checkpoint['stoi']
itos = checkpoint['itos']

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

model = GPT(cfg).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameter model")

while True:
    prompt = input("\nPrompt (empty to sample from blank, q to quit): ")
    if prompt.strip().lower() == 'q':
        break

    if prompt:
        ctx = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    else:
        ctx = torch.zeros((1, 1), dtype=torch.long, device=device)

    max_tokens = int(input("Max new tokens [500]: ") or 500)
    output = model.generate(ctx, max_new_tokens=max_tokens)
    print("\n--- Generated ---")
    print(decode(output[0].tolist()))
