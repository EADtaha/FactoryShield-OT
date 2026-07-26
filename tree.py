import os
from pathlib import Path

# Dossiers et fichiers à ignorer
IGNORE_DIRS = {'.git', 'venv', '__pycache__', '.pytest_cache', '.idea', '.vscode'}
IGNORE_FILES = {'.DS_Store'}

def print_tree(dir_path: Path, prefix: str = ''):
    contents = sorted(
        [p for p in dir_path.iterdir() if p.name not in IGNORE_DIRS and p.name not in IGNORE_FILES],
        key=lambda s: (s.is_file(), s.name.lower())
    )
    pointers = ['├── '] * (len(contents) - 1) + ['└── ']

    for pointer, path in zip(pointers, contents):
        print(f"{prefix}{pointer}{path.name}")
        if path.is_dir():
            extension = '│   ' if pointer == '├── ' else '    '
            print_tree(path, prefix=prefix + extension)

if __name__ == "__main__":
    print(f"📁 {Path.cwd().name}/")
    print_tree(Path.cwd())
