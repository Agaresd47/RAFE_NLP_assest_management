from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    local_dir="/projects/p32908/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507",
    resume_download=True,
)