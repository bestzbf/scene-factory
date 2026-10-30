import copy
import math
import unittest
from library_selection import select_and_place, LibrarySelectionError
from library_layout import expand_response
from test_library_layout import fixture
from factory_config import normalize_config
from pathlib import Path
import json

class Catalog:
    def __init__(self):
        self.data={}
        self.assets={}
        for category,dims,placement in [('table',[1.5,.8,.8],'floor'),('remote_control',[.05,.18,.02],'table')]:
            self.assets[category]={'asset_id':category,'category':category,'geometry_variant_id':category,
              'scope':'single_prop','placement':placement,'dimensions_import_units':dims,'meters_per_unit':1.,
              'front_axis':'-Y','up_axis':'+Z','material_variant_ids':[category+'_mat'],
              'scale_source':'estimated','support_surfaces':[]}
        self.assets['table']['support_surfaces']=[{'id':'top','z':.8,
             'polygon_xy':[[-.75,-.4],[.75,-.4],[.75,.4],[-.75,.4]]}]
    def production_readiness(self,asset):return True,[]
    def get_material_variant(self,identifier):return {'compatible_geometry_ids':[identifier[:-4]]}
    def verify_material_hashes(self,identifier):return True

class LibrarySelectionTests(unittest.TestCase):
    def solve(self,raw,catalog=None,options=None):
        return select_and_place(expand_response(raw),raw,catalog or Catalog(),options)
    def test_changed_table_height_recomputes_child_contact_and_keeps_input(self):
        raw=fixture();before=copy.deepcopy(raw)
        bundle,bindings,report=self.solve(raw,options={'min_uniform_scale':1,'max_uniform_scale':1})
        entries={e['id']:e for e in bundle['shared_scene']['base_scene']['objects']}
        self.assertAlmostEqual(entries['remote']['position'][2],.801)
        self.assertEqual(entries['remote']['position'][:2],[0,3])
        self.assertEqual(raw,before);self.assertEqual(report['bound_objects'],2)
        self.assertFalse(report['visual_acceptance'])
    def test_rotated_support_uses_full_child_footprint(self):
        raw=fixture();raw['objects'][1]['rotation_z_deg']=90
        raw['objects'][0]['xy']=[.38,3]
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw,options={'min_uniform_scale':1,'max_uniform_scale':1})
        self.assertEqual(caught.exception.report['errors'][0]['code'],'no_verified_support_surface')
        raw['objects'][0]['xy']=[0,3.5]
        self.assertEqual(self.solve(raw)[2]['status'],'passed')
    def test_missing_category_is_blocked_not_procedural_fallback(self):
        raw=fixture();raw['objects'][0]['category']='not_in_library'
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw)
        self.assertEqual(caught.exception.report['errors'][0]['code'],'missing_compatible_asset')
    def test_parent_without_measured_surface_is_not_assumed_bbox_top(self):
        catalog=Catalog();catalog.assets['table']['support_surfaces']=[]
        with self.assertRaises(LibrarySelectionError):self.solve(fixture(),catalog)
    def test_room_bounds_and_sibling_overlap_are_blocking(self):
        raw=fixture();raw['objects'][1]['xy']=[2.4,3];raw['objects'][0]['xy']=[2.4,3]
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw)
        self.assertIn('asset_outside_room',[e['code'] for e in caught.exception.report['errors']])
        raw=fixture();other=copy.deepcopy(raw['objects'][0]);other['id']='second';raw['objects'].append(other)
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw)
        self.assertIn('asset_bounds_overlap',[e['code'] for e in caught.exception.report['errors']])
    def test_repeated_runs_select_identical_assets(self):
        raw=fixture();catalog=Catalog()
        variant=copy.deepcopy(catalog.assets['table']);variant['asset_id']='table_b'
        catalog.assets['table_b']=variant
        a=self.solve(raw,catalog,{'seed':52});b=self.solve(raw,catalog,{'seed':52})
        self.assertEqual(a,b)
    def test_diversity_selects_different_compatible_geometry_and_material_pairs(self):
        raw=fixture();catalog=Catalog()
        raw['objects'][0]['xy']=[-1,3];raw['objects'][1]['xy']=[-1,3]
        second=copy.deepcopy(raw['objects'][1]);second.update(id='second_table',xy=[1,3])
        raw['objects'].append(second)
        variant=copy.deepcopy(catalog.assets['table'])
        variant.update(asset_id='table_b',geometry_variant_id='table_b',material_variant_ids=['table_b_mat'])
        catalog.assets['table_b']=variant
        bundle,bindings,report=self.solve(raw,catalog,{'seed':52,'diversity':True})
        pairs={(b['asset_id'],b['material_variant_id']) for b in bindings['instances'] if 'table' in b['object_id']}
        self.assertEqual(len(pairs),2)
        self.assertEqual(self.solve(raw,catalog,{'seed':52,'diversity':True}), (bundle,bindings,report))
    def test_candidates_are_verified_per_selection_not_per_object(self):
        """Readiness and material hashing walk the same candidate list for every
        object of a category. Re-verifying a candidate once per object repeats
        the expensive work without learning anything new."""
        raw=fixture()
        for index,(identifier,xx) in enumerate([('table_b',1.6),('table_c',-1.6)],start=2):
            extra=copy.deepcopy(raw['objects'][1])
            extra.update(id=identifier,xy=[xx,3])
            raw['objects'].append(extra)
        calls={'readiness':0,'materials':0}
        catalog=Catalog()
        ready,materials=catalog.production_readiness,catalog.verify_material_hashes
        def counting_ready(asset):
            calls['readiness']+=1;return ready(asset)
        def counting_materials(identifier):
            calls['materials']+=1;return materials(identifier)
        catalog.production_readiness=counting_ready
        catalog.verify_material_hashes=counting_materials
        _,_,report=self.solve(raw,catalog,{'min_uniform_scale':1,'max_uniform_scale':1})
        self.assertEqual(report['bound_objects'],4)
        # one table candidate and one remote candidate, each verified once
        self.assertEqual(calls,{'readiness':2,'materials':2})
    def test_overlap_blocks_by_default_and_settles_when_enabled(self):
        raw=fixture();other=copy.deepcopy(raw['objects'][1]);other.update(id='second',xy=[1.3,3])
        raw['objects'].append(other)
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw)
        self.assertIn('asset_bounds_overlap',[e['code'] for e in caught.exception.report['errors']])
        bundle,_,report=self.solve(raw,options={'settle_max_shift_m':.2})
        moves=report['settling']['moves_m']
        # Equal footprints: the first by ID keeps its place, the other takes the nearest free
        # spot and carries the remote resting on it.
        self.assertEqual(set(moves),{'table','remote'})
        self.assertEqual(moves['remote'],moves['table'])
        self.assertTrue(abs(moves['table'][1])<1e-9 and 0<abs(moves['table'][0])<=.2+1e-9)
        entries={e['id']:e for e in bundle['shared_scene']['base_scene']['objects']}
        gap=entries['second']['position'][0]-entries['table']['position'][0]
        self.assertGreaterEqual(gap,(entries['second']['dimensions'][0]+entries['table']['dimensions'][0])/2-1e-9)
        self.assertAlmostEqual(entries['remote']['position'][0]-entries['table']['position'][0],0)
    def test_settling_never_exceeds_its_budget(self):
        raw=fixture();other=copy.deepcopy(raw['objects'][1]);other.update(id='second',xy=[.5,3])
        raw['objects'].append(other)
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw,options={'settle_max_shift_m':.1})
        self.assertIn('asset_bounds_overlap',[e['code'] for e in caught.exception.report['errors']])
    def test_settling_puts_an_overhanging_child_back_on_its_support(self):
        raw=fixture();raw['objects'][0]['xy']=[.77,3]
        with self.assertRaises(LibrarySelectionError):self.solve(raw)
        bundle,_,report=self.solve(raw,options={'settle_max_shift_m':.1})
        remote={e['id']:e for e in bundle['shared_scene']['base_scene']['objects']}['remote']
        self.assertLessEqual(remote['position'][0]+remote['dimensions'][0]/2,.75)
        self.assertEqual(set(report['settling']['moves_m']),{'remote'})
    def test_settle_mode_accepts_a_mostly_supported_child_only(self):
        raw=fixture();raw['objects'][0]['target_dimensions_m']=[.05,.18,.02]
        catalog=Catalog();catalog.assets['remote_control']['dimensions_import_units']=[.05,.18,.02]
        catalog.assets['table']['support_surfaces'][0]['polygon_xy']=[[-.75,-.08],[.75,-.08],[.75,.08],[-.75,.08]]
        with self.assertRaises(LibrarySelectionError):self.solve(raw,catalog)
        bundle,_,report=self.solve(raw,catalog,{'settle_max_shift_m':.1})
        # The 0.8 m table is fitted inside its 0.75 m target box (scale 0.9375), so its
        # 0.16 m surface becomes 0.15 m and the 0.18 m remote rests 0.15/0.18 on it.
        self.assertAlmostEqual(report['objects']['table']['uniform_scale'],.9375,places=4)
        self.assertAlmostEqual(report['objects']['remote']['support_coverage'],.15/.18,places=4)
        catalog.assets['table']['support_surfaces'][0]['polygon_xy']=[[-.75,-.04],[.75,-.04],[.75,.04],[-.75,.04]]
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw,catalog,{'settle_max_shift_m':.1})
        self.assertEqual(caught.exception.report['errors'][0]['code'],'no_verified_support_surface')
    def small_table_catalog(self):
        # A 0.6 x 0.4 m side table: below the large-object footprint, so it may be omitted.
        catalog=Catalog();catalog.assets['table']['dimensions_import_units']=[.6,.4,.8]
        catalog.assets['table']['support_surfaces'][0]['polygon_xy']=[[-.3,-.2],[.3,-.2],[.3,.2],[-.3,.2]]
        return catalog
    def small_table_layout(self):
        raw=fixture();raw['objects'][1]['target_dimensions_m']=[.6,.4,.8]
        other=copy.deepcopy(raw['objects'][1]);other.update(id='second',xy=[.2,3])
        raw['objects'].append(other)
        # Six free remotes on 'second' make the scene 9 objects; the unplaceable table
        # and the remote resting on it are 2/9 = 22%.
        for index in range(6):
            extra=copy.deepcopy(raw['objects'][0]);extra.update(id=f'loose_{index}',xy=[-.05+.06*index,3.1],support_id='second')
            raw['objects'].append(extra)
        return raw
    def test_unplaceable_objects_are_omitted_only_within_the_limit(self):
        raw,catalog=self.small_table_layout(),self.small_table_catalog()
        for fraction in (None,.2):
            options={'settle_max_shift_m':.1,**({'settle_max_omitted_fraction':fraction} if fraction else {})}
            with self.subTest(fraction=fraction),self.assertRaises(LibrarySelectionError):self.solve(raw,catalog,options)
        bundle,bindings,report=self.solve(raw,catalog,{'settle_max_shift_m':.1,'settle_max_omitted_fraction':.25})
        omitted=report['settling']['omitted_objects']
        self.assertEqual(omitted,['remote','table'])
        self.assertEqual(report['omitted_objects'],len(omitted))
        self.assertEqual(report['settling']['large_objects'],[])
        kept={e['id'] for e in bundle['shared_scene']['base_scene']['objects']}
        self.assertFalse(kept & set(omitted))
        self.assertEqual({b['object_id'] for b in bindings['instances']},kept)
        for key in omitted:self.assertEqual(report['objects'][key]['status'],'omitted')
    def test_large_objects_are_never_omitted(self):
        # Two 1.5 x 0.8 m tables overlapping by 1 m cannot both be placed within 0.1 m.
        # Each is a large object, so the scene is blocked instead of losing one of them,
        # even though the omission budget would allow it.
        raw=fixture();other=copy.deepcopy(raw['objects'][1]);other.update(id='second',xy=[.5,3])
        raw['objects'].append(other)
        for index in range(6):
            extra=copy.deepcopy(raw['objects'][0]);extra.update(id=f'loose_{index}',xy=[.2+.1*index,2.8],support_id='second')
            raw['objects'].append(extra)
        with self.assertRaises(LibrarySelectionError) as caught:
            self.solve(raw,options={'settle_max_shift_m':.1,'settle_max_omitted_fraction':.25})
        report=caught.exception.report
        self.assertEqual(report['settling']['omitted_objects'],[])
        self.assertEqual(report['settling']['large_objects'],['second','table'])
        self.assertEqual(len(report['settling']['large_objects_unplaced']),1)
        unplaced=report['settling']['large_objects_unplaced'][0]
        self.assertEqual(report['objects'][unplaced]['omission_refused'],'large_object')
        self.assertIn('asset_bounds_overlap',[e['code'] for e in report['errors']])
    def test_requested_large_object_stays_protected_when_its_asset_is_smaller(self):
        # The model asks for 1.25 x 0.45 m counters (0.56 m2, large). The only asset is
        # 1.2 x 0.4 m (0.48 m2, small by its own size). Protection follows the request,
        # so the unplaceable counter blocks the scene instead of being omitted.
        catalog=Catalog();catalog.assets['table']['dimensions_import_units']=[1.2,.4,.8]
        catalog.assets['table']['support_surfaces'][0]['polygon_xy']=[[-.6,-.2],[.6,-.2],[.6,.2],[-.6,.2]]
        raw=fixture();raw['objects'][1]['target_dimensions_m']=[1.25,.45,.8]
        other=copy.deepcopy(raw['objects'][1]);other.update(id='second',xy=[.5,3])
        raw['objects'].append(other)
        for index in range(6):
            extra=copy.deepcopy(raw['objects'][0]);extra.update(id=f'loose_{index}',xy=[.2+.1*index,3],support_id='second')
            raw['objects'].append(extra)
        with self.assertRaises(LibrarySelectionError) as caught:
            self.solve(raw,catalog,{'settle_max_shift_m':.1,'settle_max_omitted_fraction':.25})
        settling=caught.exception.report['settling']
        self.assertEqual(settling['large_objects'],['second','table'])
        self.assertEqual(settling['omitted_objects'],[])
        self.assertEqual(len(settling['large_objects_unplaced']),1)
    def test_large_objects_may_move_past_the_normal_budget(self):
        # Two 1.5 x 0.8 m tables overlap by 0.15 m. With a 0.1 m budget a small object
        # could not clear that, but a large one gets twice the budget and is reported.
        raw=fixture();raw['objects']=[raw['objects'][1]]
        other=copy.deepcopy(raw['objects'][0]);other.update(id='second',xy=[1.35,3])
        raw['objects'].append(other)
        bundle,_,report=self.solve(raw,options={'settle_max_shift_m':.1,'min_uniform_scale':1,'max_uniform_scale':1})
        st=report['settling']
        self.assertEqual(st['omitted_objects'],[])
        self.assertEqual(st['large_object_max_shift_m'],.2)
        self.assertEqual(len(st['large_objects_moved_beyond_budget']),1)
        moved=st['large_objects_moved_beyond_budget'][0]
        self.assertTrue(.1<math.hypot(*st['moves_m'][moved])<=.2+1e-9)
    def test_large_objects_claim_their_spot_before_small_ones(self):
        # A small box overlaps the end of a large table. The table keeps the model's
        # position and the box moves, instead of the box pushing the table aside.
        raw=fixture();raw['objects']=[raw['objects'][1]]
        raw['objects'].append({'id':'box_01','category':'box','xy':[.75,3],'target_dimensions_m':[.3,.3,1.6],
                               'rotation_z_deg':0,'support_id':'floor','clearance_m':0})
        catalog=Catalog()
        catalog.assets['box']={**catalog.assets['table'],'asset_id':'box','category':'box','geometry_variant_id':'box',
                               'material_variant_ids':['box_mat'],'dimensions_import_units':[.3,.3,1.6],'support_surfaces':[]}
        bundle,_,report=self.solve(raw,catalog,{'settle_max_shift_m':.2})
        self.assertNotIn('table',report['settling']['moves_m'])
        self.assertIn('box_01',report['settling']['moves_m'])
    def test_candidate_scaled_below_the_scene_minimum_is_rejected_not_crashed(self):
        # A 12 mm mat at the forced 0.8 scale is 9.6 mm, under the 10 mm scene bound,
        # even though the 10 mm request itself is valid and within size tolerance.
        raw=fixture();raw['objects']=[raw['objects'][1]]
        raw['objects'][0]['target_dimensions_m']=[1.2,.64,.01]
        catalog=Catalog();catalog.assets['table']['dimensions_import_units']=[1.5,.8,.012]
        catalog.assets['table']['support_surfaces']=[]
        options={'min_uniform_scale':.8,'max_uniform_scale':.8}
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw,catalog,options)
        self.assertEqual(caught.exception.report['errors'][0]['code'],'missing_compatible_asset')
        self.assertEqual(caught.exception.report['objects']['table']['rejection_counts'],{'scaled_size_out_of_bounds':1})
    def test_settle_mode_drops_a_stacked_box_that_would_reach_through_the_ceiling(self):
        # Table 0.75 m + crate 0.7 m + crate 0.7 m = 2.15 m tops out at a 2.0 m ceiling.
        raw=fixture();raw['room']['height']=2.
        crate={'category':'crate','target_dimensions_m':[.6,.4,.7],'rotation_z_deg':0,'clearance_m':0,'xy':[0,3]}
        # A fourth, unrelated object makes the one unplaceable crate 1/4 of the scene.
        raw['objects']=[raw['objects'][1],{**crate,'id':'crate_low','support_id':'table'},
                        {**crate,'id':'crate_high','support_id':'crate_low'},
                        {**crate,'id':'bin_01','category':'bin','support_id':'floor','xy':[1.5,1]}]
        catalog=Catalog()
        catalog.assets['crate']={**catalog.assets['remote_control'],'asset_id':'crate','category':'crate',
            'geometry_variant_id':'crate','material_variant_ids':['crate_mat'],'dimensions_import_units':[.6,.4,.7],
            'support_surfaces':[{'id':'lid','z':.7,'polygon_xy':[[-.3,-.2],[.3,-.2],[.3,.2],[-.3,.2]]}]}
        catalog.assets['bin']={**catalog.assets['crate'],'asset_id':'bin','category':'bin','geometry_variant_id':'bin',
                               'material_variant_ids':['bin_mat'],'placement':'floor'}
        with self.assertRaises(LibrarySelectionError) as caught:self.solve(raw,catalog,{'settle_max_shift_m':.1})
        self.assertIn('asset_outside_room',[e['code'] for e in caught.exception.report['errors']])
        bundle,_,report=self.solve(raw,catalog,{'settle_max_shift_m':.1,'settle_max_omitted_fraction':.25})
        self.assertEqual(report['settling']['omitted_objects'],['crate_high'])
        tops=[e['position'][2]+e['dimensions'][2] for e in bundle['shared_scene']['base_scene']['objects']]
        self.assertLessEqual(max(tops),2.)
    def test_settle_option_is_absent_unless_configured(self):
        from auto_asset_binding import normalize_options
        self.assertNotIn('settle_max_shift_m',normalize_options({}))
        self.assertEqual(normalize_options({'settle_max_shift_m':.3})['settle_max_shift_m'],.3)
        for bad in (-.1,1.5,float('nan'),True,'0.2'):
            with self.subTest(value=bad),self.assertRaises(ValueError):normalize_options({'settle_max_shift_m':bad})
        self.assertNotIn('settle_max_omitted_fraction',normalize_options({'settle_max_shift_m':.3}))
        with self.assertRaises(ValueError):normalize_options({'settle_max_omitted_fraction':.1})
        with self.assertRaises(ValueError):normalize_options({'settle_max_shift_m':.3,'settle_max_omitted_fraction':.5})
    def test_new_profile_requires_external_auto_library_and_no_repairs(self):
        profile=json.loads((Path(__file__).parent/'config_examples/bigmodel-library-single-pass.json').read_text())
        normalize_config(profile)
        for section,key,value in [('asset_library','catalog',None),('asset_library','auto_bind',False),
                                  ('batch','no_review',False),('batch','max_repairs',1),
                                  ('pipeline','validation_retries',1),('budget','max_requests',2)]:
            invalid=copy.deepcopy(profile);invalid[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):normalize_config(invalid)
        invalid=copy.deepcopy(profile);invalid['models']['analysis']['max_retries']=1
        with self.assertRaises(ValueError):normalize_config(invalid)

if __name__=='__main__':unittest.main()

