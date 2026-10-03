"""
Create or update the Hugging Face Space that runs the voice server.

    HF_SPACE_TOKEN=hf_...  python deploy_space.py [space-name] [--private]

Needs a token with write access (huggingface.co/settings/tokens). Uploads the
server, sets a fresh VOICE_TOKEN secret if the Space has none, and prints the
URL to put in ZEN's VOICE_SERVER_URL.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    private = "--private" in sys.argv
    api = HfApi(token=os.environ["HF_SPACE_TOKEN"])
    user = api.whoami()["name"]
    repo = f"{user}/{args[0] if args else 'zen-voice'}"

    created = not api.repo_exists(repo, repo_type="space")
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    for name, source in [("README.md", HERE / "space" / "README.md"), ("Dockerfile", HERE / "Dockerfile"),
                         ("server.py", HERE / "server.py"), ("requirements.txt", HERE / "requirements.txt")]:
        api.upload_file(path_or_fileobj=str(source), path_in_repo=name, repo_id=repo, repo_type="space")

    if created:
        token = secrets.token_hex(32)
        api.add_space_secret(repo, "VOICE_TOKEN", token)
        (HERE / ".space-token").write_text(token)
        print("New VOICE_TOKEN saved to voice-server/.space-token (git-ignored). Put it in ZEN's VOICE_SERVER_TOKEN.")

    print(f"Space: https://huggingface.co/spaces/{repo}")
    print(f"VOICE_SERVER_URL=https://{repo.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
