import subprocess
import time
from pathlib import Path

# Modelli da testare
MODELS = [
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
    "mistral:7b-instruct-v0.3-q4_K_M"
]

QUERIES = [
    "Quali componenti sono isolati?",
    "Elenca i componenti collegati a R1",
    "Trova path da R1 a U1",
    "Cerca resistori da 10k",
    "Quali net hanno più di 3 componenti collegati?"
]

out_dir = Path("diagnosi_d3")
out_dir.mkdir(exist_ok=True)

for model in MODELS:
    print(f"\n====================\nTesting model: {model}\n====================")
    for i, q in enumerate(QUERIES, 1):
        print(f"  -> Q{i}: {q}")
        start = time.time()
        
        # Sostituisci i ":" con "_" per il nome file
        safe_model = model.replace(":", "_").replace(".", "_")
        out_file = out_dir / f"bench_{safe_model}_q{i}.txt"
        
        cmd = [
            "uv", "run", "schematic-extractor", "query",
            q,
            "--pdf", "test_input/bryston_schematic.pdf",
            "--model", model,
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - start
            
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Query: {q}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Time: {elapsed:.2f}s\n")
                f.write(f"Exit code: {result.returncode}\n")
                f.write("-" * 40 + "\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n" + "-" * 40 + "\n")
                f.write("STDERR:\n")
                f.write(result.stderr)
                
            print(f"     Done in {elapsed:.2f}s. Exit code: {result.returncode}")
        except subprocess.TimeoutExpired:
            print(f"     TIMEOUT after 120s")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"Query: {q}\n")
                f.write(f"Model: {model}\n")
                f.write("TIMEOUT after 120s\n")
