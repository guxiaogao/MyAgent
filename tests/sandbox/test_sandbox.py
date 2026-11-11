import pytest
import pytest_asyncio

from app.sandbox.core.sandbox import DockerSandbox, SandboxSettings


@pytest.fixture(scope="module")
def sandbox_config():
    """创建用于测试的沙箱配置。"""
    return SandboxSettings(
        image="python:3.12-slim",
        work_dir="/workspace",
        memory_limit="1g",
        cpu_limit=0.5,
        network_enabled=True,
    )


@pytest_asyncio.fixture(scope="module")
async def sandbox(sandbox_config):
    """创建并管理测试用沙箱实例。"""
    sandbox = DockerSandbox(sandbox_config)
    await sandbox.create()
    try:
        yield sandbox
    finally:
        await sandbox.cleanup()


@pytest.mark.asyncio
async def test_sandbox_working_directory(sandbox):
    """测试沙箱工作目录配置。"""
    result = await sandbox.terminal.run_command("pwd")
    assert result.strip() == "/workspace"


@pytest.mark.asyncio
async def test_sandbox_file_operations(sandbox):
    """测试沙箱文件读写操作。"""
    # 测试文件写入
    test_content = "Hello from sandbox!"
    await sandbox.write_file("/workspace/test.txt", test_content)

    # 测试文件读取
    content = await sandbox.read_file("/workspace/test.txt")
    assert content.strip() == test_content


@pytest.mark.asyncio
async def test_sandbox_python_execution(sandbox):
    """测试沙箱中的 Python 代码执行。"""
    # 写入测试文件
    await sandbox.write_file("/workspace/test.txt", "Hello from file!")

    # 写入 Python 脚本
    python_code = """
print("Hello from Python!")
with open('/workspace/test.txt') as f:
    print(f.read())
"""
    await sandbox.write_file("/workspace/test.py", python_code)

    # 执行脚本并验证输出
    result = await sandbox.terminal.run_command("python3 /workspace/test.py")
    assert "Hello from Python!" in result
    assert "Hello from file!" in result


@pytest.mark.asyncio
async def test_sandbox_file_persistence(sandbox):
    """测试沙箱中的文件持久化。"""
    # 创建多个文件
    files = {
        "file1.txt": "Content 1",
        "file2.txt": "Content 2",
        "nested/file3.txt": "Content 3",
    }

    # 写入文件
    for path, content in files.items():
        await sandbox.write_file(f"/workspace/{path}", content)

    # 验证文件内容
    for path, expected_content in files.items():
        content = await sandbox.read_file(f"/workspace/{path}")
        assert content.strip() == expected_content


@pytest.mark.asyncio
async def test_sandbox_python_environment(sandbox):
    """测试沙箱中的 Python 环境配置。"""
    # 测试 Python 版本
    result = await sandbox.terminal.run_command("python3 --version")
    assert "Python 3.10" in result

    # 测试基础模块导入
    python_code = """
import sys
import os
import json
print("Python is working!")
"""
    await sandbox.write_file("/workspace/env_test.py", python_code)
    result = await sandbox.terminal.run_command("python3 /workspace/env_test.py")
    assert "Python is working!" in result


@pytest.mark.asyncio
async def test_sandbox_network_access(sandbox):
    """测试沙箱的网络访问。"""
    if not sandbox.config.network_enabled:
        pytest.skip("Network access is disabled")

    # 测试网络连通性
    await sandbox.terminal.run_command("apt update && apt install curl -y")
    result = await sandbox.terminal.run_command("curl -I https://www.example.com")
    assert "HTTP/2 200" in result


@pytest.mark.asyncio
async def test_sandbox_cleanup(sandbox_config):
    """测试沙箱清理流程。"""
    sandbox = DockerSandbox(sandbox_config)
    await sandbox.create()

    # 创建测试文件
    await sandbox.write_file("/workspace/test.txt", "test")
    container_id = sandbox.terminal.container.id
    # 执行清理
    await sandbox.cleanup()

    # 验证容器已被移除
    import docker

    client = docker.from_env()
    containers = client.containers.list(all=True)
    assert not any(c.id == container_id for c in containers)


@pytest.mark.asyncio
async def test_sandbox_error_handling():
    """测试无效配置下的错误处理。"""
    # 测试无效配置
    invalid_config = SandboxSettings(image="nonexistent:latest", work_dir="/invalid")

    sandbox = DockerSandbox(invalid_config)
    with pytest.raises(Exception):
        await sandbox.create()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
