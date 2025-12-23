"""pytest 配置文件"""

import pytest
import sys
import os
import time

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


# ============================================================
# 进度提示 hooks
# ============================================================

_test_start_times = {}


def pytest_runtest_setup(item):
    """测试开始前的提示"""
    _test_start_times[item.nodeid] = time.time()

    # 检查是否是 GPU/slow 测试
    markers = [m.name for m in item.iter_markers()]

    if "slow" in markers or "gpu" in markers:
        print(f"\n{'='*60}")
        print(f"[GPU TEST] {item.name}")
        print(f"{'='*60}")
        print("⏳ 加载模型中... (首次运行 torch.compile 需要 5-10 分钟)")


def pytest_runtest_teardown(item, nextitem):
    """测试结束后显示耗时"""
    if item.nodeid in _test_start_times:
        elapsed = time.time() - _test_start_times[item.nodeid]
        if elapsed > 5:  # 只显示超过 5 秒的测试
            print(f"\n⏱️  {item.name} 耗时: {elapsed:.1f}s")


def pytest_report_teststatus(report, config):
    """自定义测试状态输出"""
    if report.when == "call":
        if report.passed:
            return report.outcome, "✓", "PASSED"
        elif report.failed:
            return report.outcome, "✗", "FAILED"
        elif report.skipped:
            return report.outcome, "○", "SKIPPED"
