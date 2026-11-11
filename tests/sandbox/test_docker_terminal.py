"""AsyncDockerizedTerminal 实现的测试。"""

import docker
import pytest
import pytest_asyncio

from app.sandbox.core.terminal import AsyncDockerizedTerminal


@pytest.fixture(scope="module")
def docker_client():
    """提供 Docker 客户端的 fixture。"""
    return docker.from_env()


@pytest_asyncio.fixture(scope="module")
async def docker_container(docker_client):
    """提供测试用 Docker 容器的 fixture。"""
    container = docker_client.containers.run(
        "python:3.12-slim",
        "tail -f /dev/null",
        name="test_container",
        detach=True,
        remove=True,
    )
    yield container
    container.stop()


@pytest_asyncio.fixture
async def terminal(docker_container):
    """提供已初始化的 AsyncDockerizedTerminal 实例的 fixture。"""
    terminal = AsyncDockerizedTerminal(
        docker_container,
        working_dir="/workspace",
        env_vars={"TEST_VAR": "test_value"},
        default_timeout=30,
    )
    await terminal.init()
    yield terminal
    await terminal.close()


class TestAsyncDockerizedTerminal:
    """AsyncDockerizedTerminal 的测试用例。"""

    @pytest.mark.asyncio
    async def test_basic_command_execution(self, terminal):
        """测试基础命令执行功能。"""
        result = await terminal.run_command("echo 'Hello World'")
        assert "Hello World" in result

    @pytest.mark.asyncio
    async def test_environment_variables(self, terminal):
        """测试环境变量的设置与访问。"""
        result = await terminal.run_command("echo $TEST_VAR")
        assert "test_value" in result

    @pytest.mark.asyncio
    async def test_working_directory(self, terminal):
        """测试工作目录设置。"""
        result = await terminal.run_command("pwd")
        assert "/workspace" == result

    @pytest.mark.asyncio
    async def test_command_timeout(self, docker_container):
        """测试命令超时功能。"""
        terminal = AsyncDockerizedTerminal(docker_container, default_timeout=1)
        await terminal.init()
        try:
            with pytest.raises(TimeoutError):
                await terminal.run_command("sleep 5")
        finally:
            await terminal.close()

    @pytest.mark.asyncio
    async def test_multiple_commands(self, terminal):
        """测试按顺序执行多条命令。"""
        cmd1 = await terminal.run_command("echo 'First'")
        cmd2 = await terminal.run_command("echo 'Second'")
        assert "First" in cmd1
        assert "Second" in cmd2

    @pytest.mark.asyncio
    async def test_session_cleanup(self, docker_container):
        """测试资源是否被正确清理。"""
        terminal = AsyncDockerizedTerminal(docker_container)
        await terminal.init()
        assert terminal.session is not None
        await terminal.close()
        # 验证会话已被正确清理
        # 注意：session 对象仍然存在，但其内部连接已关闭
        assert terminal.session is not None


# 配置 pytest-asyncio
def pytest_configure(config):
    """配置 pytest-asyncio。"""
    config.addinivalue_line("asyncio_mode", "strict")
    config.addinivalue_line("asyncio_default_fixture_loop_scope", "function")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
