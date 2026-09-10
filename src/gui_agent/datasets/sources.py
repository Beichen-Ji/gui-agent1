"""Canonical source URLs and licenses for supported GUI datasets."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from gui_agent.datasets.schema import DatasetSource


@dataclass(frozen=True, slots=True)
class DatasetSourceMetadata:
    """Describe the public location and license of one dataset source."""

    url: str
    license: str


DATASET_SOURCES: Mapping[DatasetSource, DatasetSourceMetadata] = MappingProxyType(
    {
        "screenagent": DatasetSourceMetadata(
            url="https://github.com/niuzaisheng/ScreenAgent",
            license="Apache-2.0 (dataset); MIT (code)",
        ),
        "mind2web": DatasetSourceMetadata(
            url="https://huggingface.co/datasets/osunlp/Mind2Web",
            license="Creative Commons Attribution 4.0 International",
        ),
        "webarena": DatasetSourceMetadata(
            url="https://github.com/web-arena-x/webarena",
            license="Apache-2.0",
        ),
    }
)
