"""Fail-closed quality contracts; subprocesses and Blender are never started."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

import batch
import quality_batch as pipeline
import visual_review as review


def scene_spec():
    return batch.read_json(Path(__file__).parent / 'examples' / 'office.json')


def report_for(spec=None, verdict='recommend_accept'):
    return {
        'schema_version': 1, 'verdict': verdict, 'summary': 'Mock visual review',
        'scores': {dimension: 95 for dimension in review.DIMENSIONS},
        'issues': [], 'asset_gaps': [], 'reference_limitations': [],
        'revised_spec': copy.deepcopy(spec or scene_spec()),
    }


def metadata():
    return {'status': 'completed', 'outputs': ['scene.blend', 'rgb.png', 'annotations.json'],
            'warning_count': 0, 'render': {'engine': 'CYCLES', 'device': 'CPU',
                                         'width': 1600, 'height': 1200, 'samples': 96}}


def write_png(path, width=1600, height=1200):
    """Small, valid losslessly encoded fixture, without image dependencies."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    header = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    pixels = (b'\x00' + b'\x80\x80\x80' * width) * height
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
                     + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class QualityGateTests(unittest.TestCase):
    def test_accepts_only_complete_high_quality_mock(self):
        result = review.quality_gate(report_for(), metadata(), {'warnings': []})
        self.assertTrue(result['accepted'])
        self.assertIn('uncalibrated', result['scope'])

    def test_model_recommendation_cannot_override_low_dimension_score(self):
        for dimension in review.DIMENSIONS:
            with self.subTest(dimension=dimension):
                report = report_for()
                report['scores'][dimension] = 89 if dimension == 'realism' else 84
                gate = review.quality_gate(report, metadata(), {'warnings': []})
                self.assertFalse(gate['accepted'])
                self.assertTrue(any(reason.startswith(dimension + ':') for reason in gate['reasons']))

    def test_high_scores_cannot_override_major_issue_or_missing_asset(self):
        for severity in ('major', 'blocking'):
            with self.subTest(severity=severity):
                report = report_for()
                report['issues'] = [{'dimension': 'geometry', 'severity': severity,
                                     'evidence': 'Foreground monitor is absent',
                                     'action': 'Restore visible monitor', 'object_ids': []}]
                self.assertFalse(review.quality_gate(report, metadata(), {'warnings': []})['accepted'])
        report = report_for()
        report['asset_gaps'] = ['Required visible furniture shape is unavailable']
        self.assertFalse(review.quality_gate(report, metadata(), {'warnings': []})['accepted'])

    def test_high_scores_cannot_override_nonacceptance_verdict(self):
        for verdict in ('repair', 'asset_limited', 'insufficient_information'):
            with self.subTest(verdict=verdict):
                self.assertFalse(review.quality_gate(report_for(verdict=verdict), metadata(), {'warnings': []})['accepted'])

    def test_missing_metadata_and_technical_validation_fail_closed(self):
        cases = [({}, {'warnings': []}), (metadata(), {}),
                 (metadata(), {'warnings': ['outside room']}),
                 ({**metadata(), 'status': 'running'}, {'warnings': []})]
        no_warning_count = metadata()
        del no_warning_count['warning_count']
        cases.append((no_warning_count, {'warnings': []}))
        for data, annotations in cases:
            with self.subTest(metadata=data, annotations=annotations):
                self.assertFalse(review.quality_gate(report_for(), data, annotations)['accepted'])

    def test_preview_size_or_sampling_cannot_pass(self):
        for width, height, samples in ((640, 480, 96), (1279, 720, 96),
                                       (1600, 719, 96), (1600, 1200, 63)):
            with self.subTest(width=width, height=height, samples=samples):
                data = metadata()
                data['render'].update(width=width, height=height, samples=samples)
                self.assertFalse(review.quality_gate(report_for(), data, {'warnings': []})['accepted'])

    def test_malformed_or_nonfinite_report_is_rejected(self):
        cases = []
        report = report_for(); del report['scores']['lighting']; cases.append(report)
        report = report_for(); report['extra'] = 'unsupported'; cases.append(report)
        for value in (float('nan'), float('inf'), -1, 101, True, '99'):
            report = report_for(); report['scores']['realism'] = value; cases.append(report)
        report = report_for(); report['revised_spec']['camera']['lens_mm'] = float('nan'); cases.append(report)
        report = report_for(); report['revised_spec']['objects'][0]['unsupported_mesh'] = 'code'; cases.append(report)
        for report in cases:
            with self.subTest(report=report):
                with self.assertRaises(ValueError):
                    review.validate_review(report)

    def test_nonfinite_or_malformed_render_metadata_cannot_pass(self):
        for field in ('width', 'height', 'samples'):
            for value in (float('nan'), float('inf'), '1600', None, True):
                with self.subTest(field=field, value=value):
                    data = metadata()
                    data['render'][field] = value
                    try:
                        gate = review.quality_gate(report_for(), data, {'warnings': []})
                    except (ValueError, TypeError):
                        continue  # Explicit rejection is also fail-closed.
                    self.assertFalse(gate['accepted'])
        data = metadata(); data['warning_count'] = None
        try:
            gate = review.quality_gate(report_for(), data, {'warnings': []})
        except (ValueError, TypeError):
            return
        self.assertFalse(gate['accepted'])


class QualityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / 'source.png'
        write_png(self.source, 16, 12)
        self.seed = self.root / 'seed.json'
        batch.atomic_json(self.seed, scene_spec())
        self.args = pipeline.parse_arguments(['--images', str(self.source), '--seed-spec', str(self.seed),
                                             '--output-dir', str(self.root / 'output'), '--max-repairs', '2'])
        self.args.blender = 'mock-blender-never-executed'
        self.job = pipeline.make_quality_job(self.source, self.args)
        self.directory = self.args.output_dir / self.job['job_id']
        self.rendered = []
        self.reviewed = []
        # review_scene requires executable discovery, but subprocess.run is always mocked below.
        self.addCleanup(patch.stopall)
        patch.object(review.shutil, 'which', return_value='mock-codex-never-executed').start()

    def fake_render(self, spec_path, iteration, job, args):
        self.rendered.append(iteration.name)
        (iteration / 'scene.blend').write_bytes(b'BLENDER-mock-file')
        write_png(iteration / 'rgb.png')
        batch.atomic_json(iteration / 'metadata.json', metadata())
        batch.atomic_json(iteration / 'annotations.json', {'warnings': []})

    def review_process(self, command, **kwargs):
        self.reviewed.append(command)
        data = json.loads(kwargs['input'].split('UNTRUSTED SCENE DATA:\n', 1)[1])
        spec = data['scene_spec']
        report = report_for(spec)
        batch.atomic_json(Path(command[command.index('-o') + 1]), report)
        return subprocess.CompletedProcess(command, 0, stdout='mock reviewer', stderr='')

    def run_mocked(self, review_side_effect=None, render_side_effect=None):
        with patch.object(pipeline, 'render_scene', side_effect=render_side_effect or self.fake_render), \
                patch.object(review.subprocess, 'run', side_effect=review_side_effect or self.review_process):
            return pipeline.run_quality_job(self.job, self.args)

    def test_review_repair_rerender_review_then_terminal_resume(self):
        def repair_once(command, **kwargs):
            result = self.review_process(command, **kwargs)
            if len(self.reviewed) == 1:
                response = Path(command[command.index('-o') + 1])
                report = batch.read_json(response)
                report['verdict'] = 'repair'
                report['scores']['framing'] = 60
                report['revised_spec']['camera']['lens_mm'] = 24
                batch.atomic_json(response, report)
            return result
        result = self.run_mocked(repair_once)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01'])
        self.assertEqual(len(self.reviewed), 2)
        self.assertEqual(len(result['history']), 2)
        self.assertEqual(result['iteration'], 'iteration_01')
        self.assertEqual(batch.read_json(self.directory / 'iteration_00' / 'scene_spec.json')['camera']['lens_mm'], 20)
        self.assertEqual(batch.read_json(self.directory / 'iteration_01' / 'scene_spec.json')['camera']['lens_mm'], 24)
        with patch.object(pipeline, 'render_scene') as render_mock, patch.object(review.subprocess, 'run') as reviewer_mock:
            resumed = pipeline.run_quality_job(self.job, self.args)
        self.assertEqual(resumed['status'], 'accepted')
        self.assertTrue(resumed['resumed'])
        render_mock.assert_not_called()
        reviewer_mock.assert_not_called()

    def test_interrupted_second_review_resumes_completed_work(self):
        def timeout_second(command, **kwargs):
            if len(self.reviewed) == 1:
                self.reviewed.append(command)
                raise subprocess.TimeoutExpired(command, 1)
            result = self.review_process(command, **kwargs)
            response = Path(command[command.index('-o') + 1])
            report = batch.read_json(response)
            report['verdict'] = 'repair'
            report['scores']['framing'] = 60
            report['revised_spec']['camera']['lens_mm'] = 24
            batch.atomic_json(response, report)
            return result
        failed = self.run_mocked(timeout_second)
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('no acceptance', failed['error'])
        self.assertFalse((self.directory / '.quality.lock').exists())
        resumed = self.run_mocked()
        self.assertEqual(resumed['status'], 'accepted')
        self.assertEqual(self.rendered, ['iteration_00', 'iteration_01'])
        self.assertEqual(len(self.reviewed), 3)

    def test_repair_budget_exhaustion_never_pretends_acceptance(self):
        def always_repair(command, **kwargs):
            result = self.review_process(command, **kwargs)
            response = Path(command[command.index('-o') + 1])
            report = batch.read_json(response)
            report['verdict'] = 'repair'
            report['scores']['realism'] = 60
            report['revised_spec']['camera']['lens_mm'] += 1
            batch.atomic_json(response, report)
            return result
        result = self.run_mocked(always_repair)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['stop_reason'], 'repair_budget_exhausted')
        self.assertEqual(len(self.rendered), self.args.max_repairs + 1)

    def test_tampered_final_artifact_cannot_reuse_acceptance(self):
        result = self.run_mocked()
        self.assertEqual(result['status'], 'accepted')
        (self.directory / 'iteration_00' / 'rgb.png').write_bytes(b'modified render')
        resumed = self.run_mocked()
        self.assertEqual(resumed['status'], 'conflict')
        self.assertIn('changed', resumed['error'])

    def test_changed_source_before_terminal_resume_cannot_reuse_acceptance(self):
        self.assertEqual(self.run_mocked()['status'], 'accepted')
        self.source.write_bytes(b'changed original reference')
        resumed = self.run_mocked()
        self.assertEqual(resumed['status'], 'conflict')

    def test_cached_review_rejects_changed_report_and_blend(self):
        self.assertEqual(self.run_mocked()['status'], 'accepted')
        iteration = self.directory / 'iteration_00'
        for filename in ('review.json', 'scene.blend'):
            with self.subTest(filename=filename):
                path = iteration / filename
                original = path.read_bytes()
                try:
                    if filename == 'review.json':
                        report = batch.read_json(path)
                        report['summary'] = 'Modified after review'
                        batch.atomic_json(path, report)
                    else:
                        path.write_bytes(b'different unreviewed Blender scene')
                    with self.assertRaises(ValueError):
                        pipeline.cached_review(self.source, iteration)
                finally:
                    path.write_bytes(original)

    def test_render_failure_and_malformed_review_cannot_produce_acceptance(self):
        def fail_render(*args):
            raise RuntimeError('mock render failed')
        result = self.run_mocked(render_side_effect=fail_render)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result.get('gate', {}).get('accepted', False))
        def invalid_review(command, **kwargs):
            response = Path(command[command.index('-o') + 1])
            response.write_text('{"verdict": "recommend_accept"}')
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        result = self.run_mocked(invalid_review)
        self.assertNotEqual(result['status'], 'accepted')
        self.assertFalse((self.directory / 'iteration_00' / 'review_provenance.json').exists())

    def test_held_lock_starts_no_work(self):
        pipeline.prepare_job(self.job, self.args.output_dir)
        (self.directory / '.quality.lock').write_text('another process')
        with patch.object(pipeline, 'render_scene') as renderer, patch.object(review.subprocess, 'run') as reviewer:
            result = pipeline.run_quality_job(self.job, self.args)
        self.assertEqual(result['status'], 'locked')
        renderer.assert_not_called()
        reviewer.assert_not_called()
        self.assertTrue((self.directory / '.quality.lock').exists())


if __name__ == '__main__':
    unittest.main()
