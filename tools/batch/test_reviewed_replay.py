"""Regression coverage for comparing independently prepared retail scenes."""
import json
from pathlib import Path
import tempfile
import unittest

from verify_retail_replay import compare


class ReviewedReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('left','right'):
            run=self.root/name;prepared=run/'prepared';render=run/'renders'/'shop'
            prepared.mkdir(parents=True);render.mkdir(parents=True)
            self.write(prepared/'preparation_report.json',{'profile_sha256':'profile','atlas_sha256':'atlas'})
            self.write(prepared/'scene_bundle.json',{'views':[{'id':'view_01','camera':{'lens_mm':21}}]})
            self.write(render/'scene_bundle.json',{'views':[{'id':'view_01','camera':{'lens_mm':21}}]})
            self.write(render/'verification.json',{'passed':True,'checked_cameras':['view_01'],
                'geometry_sha256_after_views':'same-world-geometry','stock_units':4})
            self.write(render/'retail_refinement.json',{'source':str(prepared)})
            self.write(render/'optimization_config.json',{'seed':17})

    @staticmethod
    def write(path,data):path.write_text(json.dumps(data))

    def result(self):return compare(self.root/'left/renders',self.root/'right/renders')

    def test_distinct_preparation_directories_are_repeatable(self):
        self.assertTrue(self.result()['passed'])

    def test_changed_profile_is_rejected(self):
        self.write(self.root/'right/prepared/preparation_report.json',{'profile_sha256':'other','atlas_sha256':'atlas'})
        self.assertFalse(self.result()['shops']['shop']['checks']['source_identical'])

    def test_changed_camera_is_rejected_even_when_meshes_match(self):
        self.write(self.root/'right/renders/shop/scene_bundle.json',{'views':[{'id':'view_01','camera':{'lens_mm':32}}]})
        result=self.result();self.assertFalse(result['passed'])
        self.assertTrue(result['shops']['shop']['checks']['geometry_identical'])
        self.assertFalse(result['shops']['shop']['checks']['cameras_identical'])


if __name__=='__main__':unittest.main()
