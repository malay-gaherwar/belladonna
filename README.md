# Belladonna

A research codebase to aggregate, structure, and evaluate breast-cancer knowledge, and to build a tiered benchmark for LLMs.

## Quick start
```bash
# 1) create and activate env (conda)
conda env create -f environment.yml
conda activate belladonna

# 2) install pre-commit hooks
pre-commit install

# 3) run tests + lint
make check

# 4) run a sample script
python -m belladonna --help
```

### Working with VS Code
- Open this folder in VS Code.
- Recommended extensions are auto-suggested.
- Use the built-in tasks: `Run: Lint`, `Run: Tests`, `Format`, and `Run: Sample` (Ctrl/Cmd+Shift+P → "Run Task").
- For remote GPUs (our lab "planets"), use Remote-SSH. See **Remote compute** below.

### GitHub workflow
- Commit early and often (`git add -A && git commit -m "msg"`), then `git push`.
- CI runs ruff, mypy, pytest, and pre-commit on push/PR.
- Open PRs for any non-trivial changes to keep reviewable history.

## Remote compute (planets GPUs)
1. Ensure SSH access works:
   ```bash
   ssh <username>@planet01  # replace with your host
   ```
2. In VS Code install _Remote - SSH_ extension. Add a host in your SSH config (on your laptop):
   ```
   Host planet01
     HostName planet01.example.edu
     User <username>
     IdentityFile ~/.ssh/id_ed25519
   ```
3. From VS Code: _Remote Explorer → SSH Targets → planet01 → Connect_. Clone this repo on the remote once.
4. Create the same `conda` env on the remote: `conda env create -f environment.yml && conda activate belladonna`.
5. If using a scheduler (e.g., SLURM), submit jobs with `sbatch bootstrap/slurm_example.sh`.

## Data handling
- All data lives in `data/` and is ignored by git.
- For large binary artifacts consider Git LFS (enabled via `.gitattributes`).
- If you need proper data versioning remotes, add DVC later.

## Project layout
```
belladonna/
├─ src/belladonna/             # Python package
│  ├─ __init__.py
│  ├─ __main__.py              # `python -m belladonna`
│  ├─ config.py                # pydantic-based config loader
│  └─ logging_config.py
├─ scripts/                    # CLI utilities
│  └─ download_pubmed.py
├─ notebooks/                  # Jupyter notebooks (lightweight, no secrets)
├─ configs/                    # YAML configs (env-agnostic)
│  └─ default.yaml
├─ tests/                      # pytest tests
│  └─ test_sanity.py
├─ bootstrap/                  # cluster helpers
│  └─ slurm_example.sh
├─ .github/workflows/ci.yml    # GitHub Actions
├─ .vscode/                    # VS Code settings & tasks
├─ environment.yml             # conda env
├─ pyproject.toml              # tooling config (ruff, mypy, black)
├─ .pre-commit-config.yaml
├─ .gitignore
├─ .gitattributes
├─ Makefile
└─ LICENSE
```

## Secrets
- Put API keys in environment variables or a local `.env` (not committed). Use `python-dotenv` to load if needed.
- Never commit raw credentials.
