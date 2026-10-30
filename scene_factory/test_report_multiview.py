"""Offline report scaling and path-boundary tests; no API or Blender invocation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from report_multiview import report


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / 'batch'
        self.output.mkdir()
        self.original = self.root / '原始 photo #1.png'
        Image.new('RGB', (80, 60), (40, 80, 120)).save(self.original)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def scene(self, identifier, original=None):
        folder = self.output / identifier / 'iteration_00'
        self.write_json(folder / 'scene_bundle.json', {'views': [
            {'id': 'view_01', 'source_image': str(original or self.original)}]})
        Image.new('RGB', (80, 60), (80, 120, 60)).save(folder / 'rgb_view_01.png')
        return {'scene_id': identifier, 'iteration': 'iteration_00', 'status': 'generated_unreviewed'}

    def summary(self, scenes):
        self.write_json(self.output / 'summary.json', {'scenes': scenes})

    def text(self, name='comparison.html'):
        return (self.output / name).read_text(encoding='utf-8')

    def test_small_report_and_external_original_link(self):
        self.summary([self.scene('office')])
        self.assertEqual(report(self.output), self.output / 'comparison.html')
        page = self.text()
        self.assertIn(self.original.as_uri(), page)
        self.assertIn('office/iteration_00/rgb_view_01.png', page)
        self.assertIn('office/iteration_00/scene.blend', page)
        self.assertNotIn('comparison_page_', page)
        self.assertEqual(page.count('<style>'), 1)
        self.assertEqual(page.count('</style>'), 1)
        with Image.open(self.output / 'comparison.jpg') as sheet:
            self.assertEqual(sheet.size, (1600, 640))

    def test_current_result_views_avoid_parsing_geometry_bundles(self):
        records = [self.scene(f'room_{index}') for index in range(3)]
        for record in records:
            record['views'] = [{'id': 'view_01', 'source_image': str(self.original),
                                'render': 'rgb_view_01.png', 'width': 80, 'height': 60}]
        self.summary(records)
        # A diagnostics page can use its recorded views even if the full bundle
        # is unavailable; it does not certify integrity or modify acceptance.
        for record in records:
            (self.output / record['scene_id'] / 'iteration_00/scene_bundle.json').unlink()
        from report_multiview import _read_object
        with patch('report_multiview._read_object', wraps=_read_object) as read:
            report(self.output, max_contact_views=1)
        self.assertEqual(read.call_count, 0)
        self.assertNotIn('class="report-error"', self.text())
        for record in records:
            self.assertIn(f"{record['scene_id']}/iteration_00/rgb_view_01.png", self.text())
        self.assertIn('generated_unreviewed', self.text())

    def test_invalid_summary_views_are_isolated_without_bundle_fallback(self):
        records = []
        for index, views in enumerate((None, [],
                [{'id': '../escape', 'source_image': str(self.original)}],
                [{'id': 'view_01', 'source_image': '../../../escape.png'}],
                [{'id': 'view_01', 'source_image': str(self.original)}] * 2)):
            record = self.scene(f'bad_{index}')
            record['views'] = views
            records.append(record)
        self.summary([*records, self.scene('legacy_good')])
        real_open = Image.open
        with patch('PIL.Image.open', wraps=real_open) as opened:
            report(self.output)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(self.text().count('class="report-error"'), len(records))
        self.assertIn('legacy_good/iteration_00/rgb_view_01.png', self.text())

    def test_contact_limit_prevents_decoding_all_later_images(self):
        self.summary([self.scene(f'room_{index}') for index in range(7)])
        real_open = Image.open
        with patch('PIL.Image.open', wraps=real_open) as opened:
            report(self.output, max_contact_views=2, page_size=3)
        self.assertEqual(opened.call_count, 4)  # Exactly two source/render pairs.
        self.assertIn('room_6/iteration_00/rgb_view_01.png', self.text('comparison_page_0003.html'))
        with Image.open(self.output / 'comparison.jpg') as sheet:
            self.assertEqual(sheet.size, (1600, 1280))

    def test_many_rendered_scenes_keep_default_decode_work_bounded(self):
        self.summary([self.scene(f'room_{index:03}') for index in range(123)])
        real_open = Image.open
        with patch('PIL.Image.open', wraps=real_open) as opened:
            report(self.output, max_contact_views=3)
        self.assertEqual(opened.call_count, 6)
        self.assertEqual(self.text().count('<section>'), 50)
        self.assertEqual(self.text('comparison_page_0002.html').count('<section>'), 50)
        self.assertEqual(self.text('comparison_page_0003.html').count('<section>'), 23)
        self.assertIn('room_122/iteration_00/rgb_view_01.png', self.text('comparison_page_0003.html'))

    def test_default_pagination_and_stale_pages_removed_on_shrink(self):
        scenes = [{'scene_id': f'room_{index:03}', 'status': 'pending'} for index in range(101)]
        self.summary(scenes)
        with patch('PIL.Image.open', side_effect=AssertionError('Pending scenes need no image decoding')):
            report(self.output)
        self.assertEqual(self.text().count('<section>'), 50)
        self.assertEqual(self.text('comparison_page_0002.html').count('<section>'), 50)
        self.assertEqual(self.text('comparison_page_0003.html').count('<section>'), 1)
        self.assertIn('href="comparison_page_0002.html"', self.text())
        self.assertIn('href="comparison.html"', self.text('comparison_page_0002.html'))
        self.assertIn('第 3 / 3 页', self.text('comparison_page_0003.html'))
        (self.output / 'comparison_page_notes.html').write_text('user notes')
        self.summary(scenes[:1])
        report(self.output)
        self.assertFalse((self.output / 'comparison_page_0002.html').exists())
        self.assertFalse((self.output / 'comparison_page_0003.html').exists())
        self.assertEqual(self.text('comparison_page_notes.html'), 'user notes')

    def test_ten_thousand_pending_records_produce_bounded_pages(self):
        # Scale evidence for the reporting layer only, not reconstruction quality.
        self.summary([{'scene_id': f'room_{index:05}', 'status': 'pending'} for index in range(10000)])
        with patch('PIL.Image.open', side_effect=AssertionError('No render exists')):
            report(self.output)
        pages = [self.output / 'comparison.html', *self.output.glob('comparison_page_*.html')]
        self.assertEqual(len(pages), 200)
        self.assertEqual(sum(p.read_text().count('<section>') for p in pages), 10000)
        self.assertTrue(all(p.read_text().count('<section>') <= 50 for p in pages))
        self.assertIn('room_09999', self.text('comparison_page_0200.html'))

    def test_corrupt_scene_and_missing_original_do_not_hide_valid_scene(self):
        corrupt = self.scene('corrupt')
        (self.output / 'corrupt/iteration_00/scene_bundle.json').write_text('{bad')
        missing = self.scene('missing', self.root / 'absent.png')
        malformed_review = self.scene('bad_review')
        self.write_json(self.output / 'bad_review/iteration_00/review.json', {'issues': 'invalid'})
        self.summary([corrupt, missing, malformed_review, self.scene('good'), 'invalid-record'])
        report(self.output)
        page = self.text()
        self.assertEqual(page.count('class="report-error"'), 3)
        self.assertIn('拼图预览不可用：REFERENCE', page)
        self.assertIn('good/iteration_00/rgb_view_01.png', page)
        with Image.open(self.output / 'comparison.jpg') as sheet:
            self.assertEqual(sheet.size, (1600, 1280))

    def test_metadata_text_is_escaped(self):
        self.summary([{'scene_id': 'bad<script>', 'status': 'failed', 'error': '<script>bad</script>'},
                      {'scene_id': 'safe', 'status': 'failed', 'error': '<script>bad</script>'}])
        report(self.output)
        self.assertNotIn('<script>', self.text())
        self.assertIn('&lt;script&gt;', self.text())

    def test_identifiers_and_relative_source_escape_are_isolated(self):
        bad_iteration = self.scene('bad_iteration')
        bad_iteration['iteration'] = '../../escape'
        bad_view = self.scene('bad_view')
        self.write_json(self.output / 'bad_view/iteration_00/scene_bundle.json', {'views': [
            {'id': '../../escape', 'source_image': str(self.original)}]})
        bad_source = self.scene('bad_source', '../../../escape.png')
        self.summary([{'scene_id': '../escape', 'iteration': 'iteration_00'},
                      bad_iteration, bad_view, bad_source, self.scene('good')])
        real_open = Image.open
        with patch('PIL.Image.open', wraps=real_open) as opened:
            report(self.output)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(self.text().count('class="report-error"'), 4)
        self.assertIn('good/iteration_00/rgb_view_01.png', self.text())

    def test_relative_source_inside_scene_is_resolved_from_bundle(self):
        row = self.scene('relative', '../sources/reference.png')
        source = self.output / 'relative/sources/reference.png'
        source.parent.mkdir()
        source.write_bytes(self.original.read_bytes())
        self.summary([row])
        report(self.output)
        self.assertIn('relative/sources/reference.png', self.text())
        self.assertNotIn('class="report-error"', self.text())

    def test_artifact_symlink_escape_is_not_decoded(self):
        record = self.scene('symlink')
        image = self.output / 'symlink/iteration_00/rgb_view_01.png'
        image.unlink()
        image.symlink_to(self.original)
        self.summary([record, self.scene('good')])
        real_open = Image.open
        with patch('PIL.Image.open', wraps=real_open) as opened:
            report(self.output)
        self.assertEqual(opened.call_count, 2)
        self.assertEqual(self.text().count('class="report-error"'), 1)

    def test_output_symlink_cannot_overwrite_a_different_file(self):
        self.summary([])
        protected = self.root / 'protected.txt'
        protected.write_text('keep')
        (self.output / 'comparison.html').symlink_to(protected)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            report(self.output)
        self.assertEqual(protected.read_text(), 'keep')

    def test_report_does_not_upgrade_unreviewed_status_or_write_acceptance(self):
        self.summary([self.scene('unreviewed')])
        self.write_json(self.output / 'accepted.json', [])
        before = (self.output / 'summary.json').read_bytes()
        report(self.output)
        self.assertEqual((self.output / 'summary.json').read_bytes(), before)
        self.assertEqual(json.loads((self.output / 'accepted.json').read_text()), [])
        self.assertIn('状态：generated_unreviewed', self.text())
        self.assertNotIn('本次使用自动读图和自动修正', self.text())

    def test_limits_and_invalid_summary_fail_before_report_generation(self):
        self.summary([])
        for option in ({'page_size': 0}, {'page_size': True}, {'max_contact_views': 101}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                report(self.output, **option)
        self.write_json(self.output / 'summary.json', {'scenes': None})
        with self.assertRaisesRegex(ValueError, 'scenes must be a list'):
            report(self.output)
        self.assertFalse((self.output / 'comparison.html').exists())


if __name__ == '__main__':
    unittest.main()
