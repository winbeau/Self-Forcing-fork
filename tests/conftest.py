"""pytest 配置文件"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_addoption(parser):
    """添加自定义命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (require GPU)",
    )


def pytest_configure(config):
    """配置 pytest markers"""
    config.addinivalue_line("markers", "slow: mark test as slow (require GPU)")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    """根据选项跳过测试"""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
