"""Archive reader must preserve destinations and reject unsafe/tampered data."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.fe01_a01_ops.review_archive import verify_extract


def archive(root, entries, manifest=None):
    path=root/'review.tar.gz'
    expected={name:hashlib.sha256(data).hexdigest() for name,data,kind in entries if kind==tarfile.REGTYPE}
    document=dict(source={},files_sha256=expected if manifest is None else manifest)
    with tarfile.open(path,'w:gz') as output:
        for name,data,kind in entries+[('A01_REVIEW_MANIFEST.json',json.dumps(document).encode(),tarfile.REGTYPE)]:
            member=tarfile.TarInfo(name);member.type=kind
            if kind==tarfile.SYMTYPE:member.linkname='../escape'
            member.size=len(data) if kind==tarfile.REGTYPE else 0
            output.addfile(member,io.BytesIO(data) if member.size else None)
    Path(str(path)+'.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  review.tar.gz\n')
    return path


class ArchiveTests(unittest.TestCase):
    def test_valid_manifest_and_all_members_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=archive(root,[('results/a.csv',b'1,2\n',tarfile.REGTYPE)])
            result=verify_extract(source,root/'raw')
            self.assertEqual(result['manifest_members'],1);self.assertEqual(result['regular_members'],2)
            self.assertEqual((root/'raw/results/a.csv').read_bytes(),b'1,2\n')

    def test_transport_or_member_tampering_is_rejected(self):
        for kind in ('transport','member'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source=archive(root,[('a.csv',b'1\n',tarfile.REGTYPE)],{'a.csv':'0'*64} if kind=='member' else None)
                if kind=='transport':Path(str(source)+'.sha256').write_text('0'*64+'  review.tar.gz\n')
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):verify_extract(source,root/'raw')

    def test_unsafe_paths_links_and_duplicate_members_cannot_escape_or_overwrite(self):
        variants=[('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),('a/../escape',tarfile.REGTYPE),
                  ('a//b',tarfile.REGTYPE),('link',tarfile.SYMTYPE),('duplicate',tarfile.REGTYPE)]
        for name,kind in variants:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);entries=[(name,b'a',kind)]
                if name=='duplicate':entries.append((name,b'b',kind))
                source=archive(root,entries)
                with self.assertRaisesRegex(ValueError,'Unsafe/duplicate/nonregular'):verify_extract(source,root/'raw')
                self.assertFalse((root/'escape').exists())

    def test_nonempty_or_existing_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=archive(root,[('a',b'x',tarfile.REGTYPE)])
            target=root/'raw';target.mkdir();(target/'keep').write_text('original')
            with self.assertRaises(FileExistsError):verify_extract(source,target)
            self.assertEqual((target/'keep').read_text(),'original')


if __name__=='__main__':unittest.main()
