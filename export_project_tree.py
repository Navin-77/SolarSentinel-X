from pathlib import Path

# ============================================================
# SOLARSENTINEL-X PROJECT TREE EXPORTER
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "project_tree.txt"

# Large / unnecessary folders to skip
EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    "site-packages",
}

# Files to skip
EXCLUDE_FILES = {
    "project_tree.txt",
}


def build_tree(path: Path, prefix=""):
    lines = []

    try:
        entries = []

        for item in path.iterdir():

            # Skip excluded files
            if item.name in EXCLUDE_FILES:
                continue

            # Skip excluded directories
            if item.is_dir() and item.name in EXCLUDE_DIRS:
                continue

            entries.append(item)

    except PermissionError:
        lines.append(prefix + "[Permission Denied]")
        return lines

    # Directories first, then files
    entries.sort(
        key=lambda x: (
            not x.is_dir(),
            x.name.lower()
        )
    )

    for index, item in enumerate(entries):

        is_last = index == len(entries) - 1

        if is_last:
            branch = "└── "
            next_prefix = prefix + "    "
        else:
            branch = "├── "
            next_prefix = prefix + "│   "

        if item.is_dir():

            lines.append(
                prefix + branch + item.name + "/"
            )

            lines.extend(
                build_tree(
                    item,
                    next_prefix
                )
            )

        else:

            lines.append(
                prefix + branch + item.name
            )

    return lines


def main():

    print("=" * 80)
    print("SOLARSENTINEL-X PROJECT STRUCTURE EXPORTER")
    print("=" * 80)

    tree_lines = build_tree(ROOT)

    content = []

    content.append(
        "SOLARSENTINEL-X PROJECT STRUCTURE"
    )

    content.append("=" * 80)

    content.append(
        f"Project Root: {ROOT}"
    )

    content.append("")

    content.append(
        "Excluded large/generated folders:"
    )

    content.append(
        "venv, .venv, __pycache__, .git, node_modules, etc."
    )

    content.append("")

    content.append(
        ROOT.name + "/"
    )

    content.extend(tree_lines)

    OUTPUT.write_text(
        "\n".join(content),
        encoding="utf-8"
    )

    print()
    print("Project structure exported successfully!")
    print()
    print(f"File created:")
    print(OUTPUT)
    print()
    print("Upload project_tree.txt here.")


if __name__ == "__main__":
    main()