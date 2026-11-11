import os


# 需要在文件操作中排除的文件
EXCLUDED_FILES = {
    ".DS_Store",
    ".gitignore",
    "package-lock.json",
    "postcss.config.js",
    "postcss.config.mjs",
    "jsconfig.json",
    "components.json",
    "tsconfig.tsbuildinfo",
    "tsconfig.json",
}

# 需要在文件操作中排除的目录
EXCLUDED_DIRS = {"node_modules", ".next", "dist", "build", ".git"}

# 需要在文件操作中排除的文件扩展名
EXCLUDED_EXT = {
    ".ico",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".db",
    ".sql",
}


def should_exclude_file(rel_path: str) -> bool:
    """根据路径、文件名或扩展名判断文件是否应被排除

    Args:
        rel_path: 待检查文件的相对路径

    Returns:
        应排除返回 True，否则返回 False
    """
    # 检查文件名
    filename = os.path.basename(rel_path)
    if filename in EXCLUDED_FILES:
        return True

    # 检查所在目录
    dir_path = os.path.dirname(rel_path)
    if any(excluded in dir_path for excluded in EXCLUDED_DIRS):
        return True

    # 检查扩展名
    _, ext = os.path.splitext(filename)
    if ext.lower() in EXCLUDED_EXT:
        return True

    return False


def clean_path(path: str, workspace_path: str = "/workspace") -> str:
    """清洗并归一化路径，使其相对于工作区

    Args:
        path: 待清洗的路径
        workspace_path: 要移除的工作区基准路径（默认为 "/workspace"）

    Returns:
        清洗后的路径，相对于工作区
    """
    # 移除开头斜杠
    path = path.lstrip("/")

    # 移除工作区前缀（如存在）
    if path.startswith(workspace_path.lstrip("/")):
        path = path[len(workspace_path.lstrip("/")) :]

    # 移除 workspace/ 前缀（如存在）
    if path.startswith("workspace/"):
        path = path[9:]

    # 移除剩余的开头斜杠
    path = path.lstrip("/")

    return path
