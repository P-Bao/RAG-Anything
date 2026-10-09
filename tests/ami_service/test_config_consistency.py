"""docker-compose.ami.yml và settings.py phải thống nhất về default fusion.

Compose `environment:` ghi đè default của pydantic-settings, nên nếu hai bên lệch
nhau thì container chạy một cấu hình khác với cấu hình đã đo và đề xuất. Lỗi này
đã xảy ra nhiều lần trong quá trình chọn fusion mode: dòng bench trông hợp lý
nhưng thực ra đang đo cấu hình cũ.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from ami_rag.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.ami.yml"

# Biến trong compose có default `:-` và cần khớp với settings.py.
SYNCED_KEYS = [
    "RETRIEVAL_FUSION_MODE",
    "RETRIEVAL_FUSION_POOLS",
    "RETRIEVAL_FUSION_POOL_SIZES",
    "RETRIEVAL_FUSION_RRF_K",
    "RETRIEVAL_FUSION_QUOTA",
    "RETRIEVAL_FUSION_IMAGE_GATE",
    "RETRIEVAL_FUSION_VL_POOLS",
]


def _compose_environment() -> dict:
    data = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
    return data["services"]["ami-rag-api"]["environment"]


def _compose_default(key: str) -> str:
    """Lấy giá trị fallback trong `${KEY:-default}`."""
    line = _compose_environment()[key]
    match = re.search(r"\$\{\w+:-(.*)\}\s*$", line)
    assert match, f"{key} không có default dạng ${{...:-...}}: {line!r}"
    return match.group(1)


def _settings_default(key: str) -> str:
    return str(Settings.model_fields[key].default)


def _comparable(value: str) -> str:
    """So sánh không phân biệt hoa thường: compose luôn ghi `true`, settings là `True`."""
    return value.strip().lower()


@pytest.mark.parametrize("key", SYNCED_KEYS)
def test_compose_default_matches_settings(key: str) -> None:
    assert _comparable(_compose_default(key)) == _comparable(_settings_default(key)), (
        f"{key}: docker-compose.ami.yml đặt {_compose_default(key)!r} nhưng "
        f"settings.py đặt {_settings_default(key)!r}. Compose ghi đè default của "
        f"pydantic-settings nên container sẽ chạy giá trị của compose."
    )


def test_every_interpolated_fusion_key_is_asserted() -> None:
    """Mọi biến fusion được interpolate trong compose đều phải được kiểm."""
    interpolated = {
        key
        for key, value in _compose_environment().items()
        if key.startswith("RETRIEVAL_FUSION_") and ":-" in value
    }
    assert interpolated == set(SYNCED_KEYS), (
        f"Biến fusion trong compose chưa có test đồng bộ: {sorted(interpolated - set(SYNCED_KEYS))}"
    )