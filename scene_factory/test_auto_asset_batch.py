"""Automatic selection in the grouped runner; mocked API/render, real catalog rules."""
import copy
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import batch_multiview as runner
import vision_multiview as vision
from batch import atomic_json, read_json
import test_auto_asset_binding as binding_tests
import test_batch_multiview as batch_tests


class AutomaticAssetBatchTests(unittest.TestCase):
    def setUp(self):
        batch_tests.GroupedBatchTests.setUp(self)
        fixture = binding_tests.AutoAssetBindingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.library = fixture
        self.catalog_path = fixture.root / 'catalog.json'
        self.save_catalog()
        self.args.asset_catalog = self.catalog_path
        self.args.asset_auto_bind = True
        self.args.asset_preview = False
        self.args.asset_selection_options = {'seed': 31}

    def save_catalog(self):
        atomic_json(self.catalog_path, self.library.data)

    def fake_analyze(self, scene_id, views, output, **kwargs):
        bundle = batch_tests.bundle_for(scene_id, views)
        bundle['shared_scene']['base_scene']['objects'] = [copy.deepcopy(self.library.object())]
        atomic_json(output, bundle)
        atomic_json(output.parent / 'analysis_provenance.json',
                    {'bundle_sha256': runner.file_sha256(output)})
        return bundle

    fake_render = batch_tests.GroupedBatchTests.fake_render

    def review_report(self, bundle, *, score=70, verdict='repair', revised=None):
        """Valid independent score fixture for the real candidate comparator."""
        scores = {dimension: score for dimension in vision.DIMENSIONS}
        return {'schema_version': 3, 'summary': 'Synthetic asset selection review.',
                'scores': scores, 'verdict': verdict,
                'per_view_scores': [{'view_id': view['id'], 'scores': dict(scores), 'summary': 'Synthetic fixture.'}
                                    for view in bundle['views']],
                'issues': [], 'asset_gaps': [], 'reference_limitations': [],
                'revised_bundle': copy.deepcopy(revised if revised is not None else bundle)}

    def run_mocked(self, group=None, renderer=None):
        group = group or runner.load_groups(self.groups_path)[0]
        with patch.object(vision, 'analyze', side_effect=self.fake_analyze) as analyze, \
             patch.object(runner, 'run_stage', side_effect=renderer or self.fake_render) as render:
            result = runner.run_job(group, self.args, self.pipeline)
        return result, analyze, render

    def test_standalone_asset_job_respects_batch_lock(self):
        group = runner.load_groups(self.groups_path)[0]
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        with runner.process_lock(self.args.output_dir / '.batch.lock'):
            result, analyze, render = self.run_mocked(group)
        self.assertEqual(result['status'], 'locked')
        self.assertEqual((analyze.call_count, render.call_count), (0, 0))

    def test_batch_asset_scope_restores_marker_after_exit_and_error(self):
        args = copy.copy(self.args)
        self.assertFalse(hasattr(args, '_asset_batch_scope_active'))
        with runner._batch_asset_scope(args):
            self.assertTrue(args._asset_batch_scope_active)
        self.assertFalse(hasattr(args, '_asset_batch_scope_active'))

        args._asset_batch_scope_active = False
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with runner._batch_asset_scope(args):
                self.assertTrue(args._asset_batch_scope_active)
                raise RuntimeError('fixture failure')
        self.assertIs(args._asset_batch_scope_active, False)

    def test_config_asset_paths_resolve_from_config_directory_after_chdir(self):
        config_dir = self.root / 'configuration'
        config_dir.mkdir()
        other_cwd = self.root / 'unrelated_working_directory'
        other_cwd.mkdir()
        config_path = config_dir / 'factory.json'
        previous_cwd = Path.cwd()
        for automatic in (True, False):
            with self.subTest(auto_bind=automatic):
                library_dir = config_dir / 'library'
                library_dir.mkdir(exist_ok=True)
                catalog_data = copy.deepcopy(self.library.data)
                catalog_data['root'] = str(self.library.root)
                atomic_json(library_dir / 'catalog.json', catalog_data)
                atomic_json(config_path, {
                    'schema_version': 1,
                    'models': {'analysis': {
                        'provider': 'openai_chat',
                        'base_url': 'https://models.example.test/v1',
                        'model': 'offline-fixture',
                        'api_key_env': 'SCENE_TEST_UNUSED_KEY',
                    }},
                    'batch': {'workers': 1, 'no_review': True},
                    'asset_library': {
                        'catalog': 'library/catalog.json',
                        'bindings': None if automatic else 'bindings/scene.json',
                        'auto_bind': automatic,
                    },
                })
                try:
                    os.chdir(other_cwd)
                    with patch('model_api.preflight') as preflight, \
                         patch.object(runner, 'discover_blender', return_value='offline-blender'), \
                         patch.object(runner, 'pipeline_hashes', return_value={}), \
                         patch.object(runner, 'run_job', side_effect=lambda group, args, hashes: {
                             'scene_id': group['scene_id'], 'status': 'generated_unreviewed',
                         }) as run_job, \
                         patch('report_multiview.report'), redirect_stdout(io.StringIO()):
                        status = runner.main([
                            '--config', '../configuration/factory.json',
                            '--groups', str(self.groups_path),
                            '--output-dir', str(self.root / f'paths_{automatic}'),
                        ])
                finally:
                    os.chdir(previous_cwd)
                self.assertEqual(status, 0)
                preflight.assert_called_once()
                self.assertEqual(run_job.call_count, 3)
                for call in run_job.call_args_list:
                    args = call.args[1]
                    self.assertEqual(args.asset_catalog, (config_dir / 'library/catalog.json').resolve())
                    self.assertEqual(args.asset_auto_bind, automatic)
                    expected = None if automatic else (config_dir / 'bindings/scene.json').resolve()
                    self.assertEqual(args.asset_bindings, expected)

    def test_each_scene_gets_its_own_bindings_shared_by_all_cameras(self):
        self.library.object()['repeat'] = [2, 1]
        groups = runner.load_groups(self.groups_path)
        paths = []
        for group in groups[:2]:
            result, _, render = self.run_mocked(group)
            self.assertEqual(result['status'], 'generated_unreviewed')
            command = render.call_args.args[0]
            path = Path(command[command.index('--asset-bindings') + 1])
            paths.append(path)
            self.assertEqual(path.parent, (self.args.output_dir / group['scene_id'] / 'iteration_00').resolve())
            self.assertEqual(command[command.index('--asset-catalog') + 1], str(self.catalog_path.resolve()))
            self.assertEqual([b['object_id'] for b in read_json(path)['instances']],
                             ['chair__0_0', 'chair__1_0'])
            state = read_json(path.parent / 'render_state.json')
            self.assertTrue({'asset_bindings.json', 'asset_selection.json'} <= set(state['artifacts']))
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(len(read_json(paths[0].parent / 'scene_bundle.json')['views']), 2)

    def test_draft_candidates_fall_back_with_explicit_evidence(self):
        self.library.data['assets'][0]['status'] = 'draft'
        self.save_catalog()
        result, _, _ = self.run_mocked()
        self.assertEqual(result['status'], 'generated_unreviewed')
        folder = Path(result['directory']) / result['iteration']
        self.assertEqual(read_json(folder / 'asset_bindings.json')['instances'], [])
        report = read_json(folder / 'asset_selection.json')
        self.assertEqual(report['fallback_objects'], 1)
        self.assertIn('asset_not_approved', report['objects']['chair']['candidates'][0]['reasons'])

    def test_repairs_reselect_size_and_do_not_retain_removed_ids(self):
        second = self.library.asset('chair_02')
        second['dimensions_import_units'] = [.7, .84, 1.26]
        self.library.data['assets'].append(second)
        self.library.data['material_variants']['mat_01']['compatible_geometry_ids'].append('chair_02')
        self.save_catalog()
        self.args.no_review, self.args.max_repairs = False, 2

        def review(bundle, directory, **kwargs):
            revised = copy.deepcopy(bundle)
            index = int(directory.name.rsplit('_', 1)[1])
            if index == 0:
                revised['shared_scene']['base_scene']['objects'][0]['dimensions'] = [.7, .84, 1.26]
            else:
                revised['shared_scene']['base_scene']['objects'][0].update(id='replacement', category='shelf')
            # Each candidate deliberately improves so this test isolates
            # asset reselection, not the separately tested rollback policy.
            return self.review_report(bundle, score=60 + index * 10, revised=revised)

        with patch.object(vision, 'review', side_effect=review), \
             patch.object(vision, 'quality_gate', return_value={'accepted': False, 'reasons': []}):
            result, _, render = self.run_mocked()
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(render.call_count, 3)
        bindings = [read_json(Path(result['directory']) / f'iteration_{i:02d}' / 'asset_bindings.json')
                    for i in range(3)]
        self.assertEqual(bindings[0]['instances'][0]['asset_id'], 'chair_01')
        self.assertEqual(bindings[1]['instances'][0]['asset_id'], 'chair_02')
        self.assertEqual(bindings[2]['instances'], [])
        last = read_json(Path(result['directory']) / 'iteration_02' / 'asset_selection.json')
        self.assertEqual(set(last['objects']), {'replacement'})

    def test_auto_selection_resume_is_deterministic(self):
        first, _, _ = self.run_mocked()
        second, analyze, render = self.run_mocked()
        self.assertEqual(first['status'], 'generated_unreviewed')
        self.assertTrue(second['resumed'])
        self.assertEqual((analyze.call_count, render.call_count), (0, 0))

    def test_tampered_selection_cannot_reuse_render(self):
        result, _, _ = self.run_mocked()
        atomic_json(Path(result['directory']) / result['iteration'] / 'asset_selection.json', {})
        resumed, analyze, render = self.run_mocked()
        self.assertIn(resumed['status'], ('failed', 'conflict'))
        self.assertIn('changed or missing', resumed['error'])
        self.assertEqual((analyze.call_count, render.call_count), (0, 0))

    def add_unused_candidate(self, missing=False):
        asset = self.library.asset('unused')
        asset.update(category='table', model_file='unused.glb')
        if not missing:
            (self.library.root / 'unused.glb').write_bytes(b'unused original')
            asset['hash'] = self.library.digest('unused.glb')
        self.library.data['assets'].append(asset)
        self.save_catalog()

    def test_unselected_candidate_changes_invalidate_resume(self):
        self.add_unused_candidate()
        self.run_mocked()
        (self.library.root / 'unused.glb').write_bytes(b'changed')
        result, analyze, render = self.run_mocked()
        self.assertIn(result['status'], ('failed', 'conflict'))
        self.assertIn('Asset library changed since this batch was started', result['error'])
        self.assertEqual((analyze.call_count, render.call_count), (0, 0))

    def test_previously_missing_candidate_appearing_invalidates_resume(self):
        self.add_unused_candidate(missing=True)
        self.run_mocked()
        (self.library.root / 'unused.glb').write_bytes(b'new candidate')
        result, analyze, render = self.run_mocked()
        self.assertIn(result['status'], ('failed', 'conflict'))
        self.assertEqual((analyze.call_count, render.call_count), (0, 0))

    def test_library_change_during_analysis_stops_before_render(self):
        self.add_unused_candidate()
        original = self.fake_analyze
        def analyze(*args, **kwargs):
            bundle = original(*args, **kwargs)
            (self.library.root / 'unused.glb').write_bytes(b'changed in analysis')
            return bundle
        self.fake_analyze = analyze
        result, _, render = self.run_mocked()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Asset library input changed', result['error'])
        self.assertEqual(render.call_count, 0)

    def test_preview_can_never_be_accepted_even_when_reviewer_accepts(self):
        self.library.data['assets'][0]['status'] = 'draft'
        self.save_catalog()
        self.args.asset_preview, self.args.no_review = True, False
        with patch.object(vision, 'review', side_effect=lambda bundle, directory, **kwargs:
                          self.review_report(bundle, score=95, verdict='recommend_accept')), \
             patch.object(vision, 'quality_gate', return_value={'accepted': True, 'reasons': []}):
            result, _, render = self.run_mocked()
        self.assertEqual(result['status'], 'needs_review')
        self.assertIn('draft_asset_preview_only', result['gate']['reasons'])
        self.assertIn('--asset-preview', render.call_args.args[0])

    def test_snapshot_backed_accepted_resume_reuses_library_hash_check(self):
        self.args.no_review, self.args.max_repairs = False, 0
        with patch.object(vision, 'review', side_effect=lambda bundle, directory, **kwargs:
                          self.review_report(bundle, score=95, verdict='recommend_accept')), \
             patch.object(vision, 'quality_gate', return_value={'accepted': True, 'reasons': []}):
            first, _, _ = self.run_mocked()
            self.assertEqual(first['status'], 'accepted')
            with patch.object(runner, 'source_check', wraps=runner.source_check) as check, \
                 patch.object(runner, 'verify_selected_asset_inputs',
                              side_effect=AssertionError('snapshot-backed resume must not reread selected assets')), \
                 patch.object(vision, 'analyze', side_effect=AssertionError('accepted analysis should resume')), \
                 patch.object(runner, 'run_stage', side_effect=AssertionError('accepted render should resume')):
                resumed, _, _ = self.run_mocked()
        self.assertTrue(resumed['resumed'])
        self.assertEqual(resumed['status'], 'accepted')
        self.assertGreaterEqual(check.call_count, 1)


if __name__ == '__main__':
    unittest.main()
