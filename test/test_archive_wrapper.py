import numpy as np
import pytest
from algatross.quality_diversity.archives.unstructured import UnstructuredArchive
from algatross.quality_diversity.archives.dask_array_store import ArrayStore
from ribs.archives import GridArchive, CVTArchive, SlidingBoundariesArchive
import dask.array as da


@pytest.fixture(params=["GridArchive", "CVTArchive", "SlidingBoundariesArchive", "UnstructuredArchive"])
def get_archive(request, tmp_path):
    seed = 1000
    extra_fields = {"field_1": ((2, 2), np.float32), "field_2": ((1,), np.uint8)}
    if request.param == "GridArchive":
        archive = GridArchive(solution_dim=3, dims=[2] * 3, ranges=[(-1, 1)] * 3, extra_fields=extra_fields, seed=seed)
    elif request.param == "CVTArchive":
        archive = CVTArchive(solution_dim=3, cells=8, ranges=[(-1, 1)] * 3, extra_fields=extra_fields, seed=seed)
    elif request.param == "SlidingBoundariesArchive":
        archive = SlidingBoundariesArchive(solution_dim=3, dims=[2] * 3, ranges=[(-1, 1)] * 3, extra_fields=extra_fields, seed=seed)
    elif request.param == "UnstructuredArchive":
        archive = UnstructuredArchive(
            solution_dim=3, measure_dim=3, k_neighbors=3, novelty_threshold=0.01, extra_fields=extra_fields, seed=seed
        )
    else:
        raise ValueError(request.param)
    archive._store = ArrayStore.from_raw_dict(archive._store.as_raw_dict(), storage_path=tmp_path)
    archive.add(
        [(0.0, 0.0, i) for i in range(8)],
        np.arange(8),
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)],
        field_1=[np.arange(i, i + 2, 0.5).reshape(2, 2) for i in range(8)],
        field_2=[np.array([str(i)]) for i in range(8)],
    )
    return archive


def test_archive_attributes(get_archive):
    """Test that the attributes of the wrapped archive are properly accessible."""
    assert get_archive.cells == 8
    assert get_archive.solution_dim == 3
    assert get_archive.measure_dim == 3
    assert get_archive.learning_rate == 1.0
    assert get_archive.threshold_min == -np.inf
    assert get_archive.qd_score_offset == 0.0
    assert not get_archive.empty
    assert len(get_archive) == 8
    assert all(field in get_archive.field_list for field in ["solution", "objective", "measures", "threshold", "field_1", "field_2"])

    # make sure all the fields are correct
    [retrieved, fields], *_ = da.compute(get_archive.retrieve([(-1, -1, -1), (1, 1, 1)]))
    assert all(retrieved)
    for key, value in [
        ("solution", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 7.0]])),
        ("objective", np.array([0.0, 7.0])),
        ("field_1", np.array([[[0.0, 0.5], [1.0, 1.5]], [[7.0, 7.5], [8.0, 8.5]]])),
        ("field_2", np.array([["0"], ["7"]], dtype=np.uint8)),
    ]:
        assert np.all(fields[key] == value)

    # check that the archive is using files instead of storing in memory
    for field in get_archive.field_list:
        assert (get_archive._store.storage_path / "fields" / field).exists()
