import asyncio
import os
import tempfile
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.sandbox.core.manager import SandboxManager


@pytest_asyncio.fixture(scope="function")
async def manager() -> AsyncGenerator[SandboxManager, None]:
    """创建沙箱管理器实例。

    使用 function 作用域，确保每个测试用例拥有独立的管理器实例。
    """
    manager = SandboxManager(max_sandboxes=2, idle_timeout=60, cleanup_interval=30)
    try:
        yield manager
    finally:
        # 确保清理所有资源
        await manager.cleanup()


@pytest.fixture
def temp_file():
    """创建临时测试文件。"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("test content")
        path = f.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)


@pytest.mark.asyncio
async def test_create_sandbox(manager):
    """测试沙箱创建。"""
    # 创建默认沙箱
    sandbox_id = await manager.create_sandbox()
    assert sandbox_id in manager._sandboxes
    assert sandbox_id in manager._last_used

    # 验证沙箱功能
    sandbox = await manager.get_sandbox(sandbox_id)
    result = await sandbox.run_command("echo 'test'")
    assert result.strip() == "test"


@pytest.mark.asyncio
async def test_max_sandboxes_limit(manager):
    """测试沙箱数量上限限制。"""
    created_sandboxes = []
    try:
        # 创建最大数量的沙箱
        for _ in range(manager.max_sandboxes):
            sandbox_id = await manager.create_sandbox()
            created_sandboxes.append(sandbox_id)

        # 验证已创建的沙箱数量
        assert len(manager._sandboxes) == manager.max_sandboxes

        # 尝试创建超出上限的沙箱应当失败
        with pytest.raises(RuntimeError) as exc_info:
            await manager.create_sandbox()

        # 验证错误消息
        expected_message = (
            f"Maximum number of sandboxes ({manager.max_sandboxes}) reached"
        )
        assert str(exc_info.value) == expected_message

    finally:
        # 清理所有已创建的沙箱
        for sandbox_id in created_sandboxes:
            try:
                await manager.delete_sandbox(sandbox_id)
            except Exception as e:
                print(f"清理沙箱 {sandbox_id} 失败：{e}")


@pytest.mark.asyncio
async def test_get_nonexistent_sandbox(manager):
    """测试获取不存在的沙箱。"""
    with pytest.raises(KeyError, match="Sandbox .* not found"):
        await manager.get_sandbox("nonexistent-id")


@pytest.mark.asyncio
async def test_sandbox_cleanup(manager):
    """测试沙箱删除清理功能。"""
    sandbox_id = await manager.create_sandbox()
    assert sandbox_id in manager._sandboxes

    await manager.delete_sandbox(sandbox_id)
    assert sandbox_id not in manager._sandboxes
    assert sandbox_id not in manager._last_used


@pytest.mark.asyncio
async def test_idle_sandbox_cleanup(manager):
    """测试空闲沙箱的自动清理。"""
    # 设置较短的空闲超时
    manager.idle_timeout = 0.1

    sandbox_id = await manager.create_sandbox()
    assert sandbox_id in manager._sandboxes

    # 等待超过空闲超时时间
    await asyncio.sleep(0.2)

    # 触发清理
    await manager._cleanup_idle_sandboxes()
    assert sandbox_id not in manager._sandboxes


@pytest.mark.asyncio
async def test_manager_cleanup(manager):
    """测试管理器的整体清理功能。"""
    # 创建多个沙箱
    sandbox_ids = []
    for _ in range(2):
        sandbox_id = await manager.create_sandbox()
        sandbox_ids.append(sandbox_id)

    # 清理所有资源
    await manager.cleanup()

    # 验证所有沙箱均已被清理
    assert not manager._sandboxes
    assert not manager._last_used


if __name__ == "__main__":
    pytest.main(["-v", __file__])
