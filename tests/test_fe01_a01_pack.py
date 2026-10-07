import json
import tarfile
import pytest
from fe01_a01.pack import review_package


def test_review_pack_excludes_weights_arrays_and_redacts_private_paths(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps({'data':{'/public/home/private/data/japan_2000.hdf5':{'bytes':10}},
        'encoder':'/public/home/private/model.pt'}))
    (tmp_path/'best.pth').write_bytes(b'synthetic excluded weight bytes')
    trace=tmp_path/'traces';trace.mkdir();(trace/'array.npz').write_bytes(b'synthetic excluded trace bytes')
    review_package(tmp_path)
    with tarfile.open(tmp_path/'review.tar.gz') as archive:
        assert 'best.pth' not in archive.getnames()
        assert not any('traces/' in n for n in archive.getnames())
        content=archive.extractfile('metadata.json').read().decode()
        assert '/public/home' not in content and 'PRIVATE_PATH/japan_2000.hdf5' in content
        assert 'A01_REVIEW_MANIFEST.json' in archive.getnames()
    assert (tmp_path/'review.tar.gz.sha256').exists()
    with pytest.raises(ValueError,match='exists'):review_package(tmp_path)


def test_source_pack_locates_namespace_dependency_without___file__():
    from scripts.fe01_a01.pack_source import dependency_root
    path=dependency_root()
    assert path.is_dir() and (path/'training/modeling.py').is_file()
