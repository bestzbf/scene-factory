import copy
import unittest
from camera_alignment import fit_camera,parameters,project,align
from test_library_layout import fixture
from library_layout import expand_response,validate_layout,canonicalize_model_layout


class CameraAlignmentTests(unittest.TestCase):
    def test_bad_optional_hints_are_dropped_without_losing_a_valid_shared_scene(self):
        raw=fixture();before=copy.deepcopy(raw)
        anchor={'view_id':'view_01','object_id':'table','corner':'top_center','image_xy':[.4,.5],'confidence':.85}
        raw['camera_anchors']=[anchor,{**anchor,'confidence':.9},{**anchor,'object_id':'missing'},
                               {**anchor,'corner':'bottom_center'},
                               {**anchor,'corner':'bottom_center','image_xy':[.5,.6]}]
        cleaned,changes=canonicalize_model_layout(raw)
        self.assertEqual(cleaned['camera_anchors'],[anchor])
        self.assertEqual(cleaned['objects'],before['objects'])
        self.assertEqual(cleaned['views'],before['views'])
        self.assertEqual(len(changes),3)
        validate_layout(cleaned)
        self.assertEqual(len(raw['camera_anchors']),5)

    def test_known_perspective_is_recovered_without_moving_geometry(self):
        truth={'position':[.2,.6,2.5],'target':[0,4,1.0],'lens_mm':24,'roll_degrees':1}
        world=[[-1,2,.2],[1,2,.2],[-1,2,1.6],[1,2,1.6],[-1.2,4,.1],[1.2,4,.1],[-1.2,4,1.8],[1.2,4,1.8]]
        pixels=project(world,parameters(truth),1280,720)[0]/[1280,720]
        initial=copy.deepcopy(truth);initial['position']=[.35,.5,2.4];initial['target']=[.15,4,1.2];initial['lens_mm']=25
        fitted,record=fit_camera(initial,world,pixels,[.95]*8,{'width':6,'depth':8,'height':3.2},1280,720)
        self.assertEqual(record['status'],'applied')
        self.assertLess(record['after_median_px'],1)
        self.assertIsNotNone(fitted)

    def test_sparse_or_unknown_landmarks_are_not_applied(self):
        raw=fixture();raw['camera_anchors']=[{'view_id':'view_01','object_id':'table','corner':'top_center',
                                              'image_xy':[.5,.5],'confidence':.9}]
        bundle=expand_response(raw);fitted,report=align(bundle,raw)
        self.assertEqual(bundle,fitted);self.assertEqual(report['view_01']['status'],'skipped')
        raw['camera_anchors'][0]['object_id']='unknown'
        with self.assertRaisesRegex(ValueError,'camera anchor'):validate_layout(raw)


if __name__=='__main__':unittest.main()
