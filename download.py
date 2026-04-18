import sys, os
print("executable:", sys.executable)
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("PYTHONHOME:", os.environ.get("PYTHONHOME"))
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("sys.path:")
for p in sys.path:
    print(" ", p)


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-9B-Base",
    cache_dir="/scratch/xla2767/hold2/hf_cache",
)

snapshot_download(
    repo_id="Skywork/Skywork-Reward-V2-Qwen3-4B",
    cache_dir="/scratch/xla2767/hold2/hf_cache",
)
