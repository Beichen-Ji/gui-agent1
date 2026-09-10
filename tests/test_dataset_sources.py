from gui_agent.datasets.sources import DATASET_SOURCES, DatasetSourceMetadata


def test_dataset_source_metadata_preserves_urls_and_licenses() -> None:
    assert DATASET_SOURCES["screenagent"] == DatasetSourceMetadata(
        url="https://github.com/niuzaisheng/ScreenAgent",
        license="Apache-2.0 (dataset); MIT (code)",
    )
    assert DATASET_SOURCES["mind2web"].license == (
        "Creative Commons Attribution 4.0 International"
    )
    assert DATASET_SOURCES["webarena"].url == (
        "https://github.com/web-arena-x/webarena"
    )
