import base64
from dataclasses import dataclass

import requests

from src import config

_GITHUB_API = "https://api.github.com"


@dataclass
class RepoFile:
    path: str
    content: str


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get(url: str) -> dict:
    response = requests.get(url, headers=_auth_headers())
    if response.status_code == 401:
        raise RuntimeError(
            "GitHub authentication failed: GITHUB_TOKEN is invalid or expired"
        )
    if response.status_code == 404:
        raise RuntimeError(
            f"GitHub resource not found: {url} — verify GITHUB_REPO is correct"
            " and the token has read access to that repository"
        )
    response.raise_for_status()
    return response.json()


def fetch_python_files() -> list[RepoFile]:
    parts = config.GITHUB_REPO.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"GITHUB_REPO must be in 'owner/name' format, got: {config.GITHUB_REPO!r}"
        )
    owner, repo = parts

    repo_info: dict = _get(f"{_GITHUB_API}/repos/{owner}/{repo}")
    default_branch: str = repo_info["default_branch"]

    tree_response: dict = _get(
        f"{_GITHUB_API}/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
    )

    py_entries: list[dict] = [
        entry
        for entry in tree_response.get("tree", [])
        if entry["type"] == "blob" and entry["path"].endswith(".py")
    ]

    if not py_entries:
        return []

    files: list[RepoFile] = []
    for entry in py_entries:
        file_response: dict = _get(
            f"{_GITHUB_API}/repos/{owner}/{repo}/contents/{entry['path']}"
        )
        decoded_content: str = base64.b64decode(file_response["content"]).decode("utf-8")
        files.append(RepoFile(path=entry["path"], content=decoded_content))

    return files
