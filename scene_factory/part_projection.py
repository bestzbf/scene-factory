"""Evaluated per-part camera projections, explicitly excluding occlusion tests."""
from __future__ import annotations

import math

import bpy


def _clip_polygon(points, axis, limit, keep_greater):
    """Sutherland-Hodgman half-space clipping for convex triangles/polygons."""
    if not points:
        return []
    result = []
    previous = points[-1]
    previous_inside = previous[axis] >= limit if keep_greater else previous[axis] <= limit
    for point in points:
        inside = point[axis] >= limit if keep_greater else point[axis] <= limit
        if inside != previous_inside:
            fraction = (limit - previous[axis]) / (point[axis] - previous[axis])
            result.append(tuple(previous[i] + fraction * (point[i] - previous[i]) for i in range(len(point))))
        if inside:
            result.append(point)
        previous, previous_inside = point, inside
    return result


def _area(points):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) * .5 if len(points) >= 3 else 0


def _round(values):
    return [round(float(value), 7) for value in values]


def _project_part(obj, camera, projection, depsgraph):
    result = {'world_aabb_m': None, 'camera_depth_range_m': None,
              'front_vertex_fraction': 0, 'all_vertices_in_front': False,
              'projected_bbox_normalized': None, 'frame_clipped_bbox_normalized': None,
              'projected_bbox_in_frame_fraction': 0, 'frame_bbox_coverage_fraction': 0,
              'intersects_frame_before_occlusion': False, 'status': 'missing_geometry',
              'render_enabled': not obj.hide_render and bool(getattr(obj, 'visible_camera', True))}
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh or not mesh.vertices or not mesh.polygons:
            return result
        world = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        inverse = camera.matrix_world.inverted()
        camera_points = [inverse @ point for point in world]
        depth = [-point.z for point in camera_points]
        if not all(math.isfinite(value) for point in camera_points for value in point):
            raise ValueError('Non-finite evaluated custom geometry cannot be projected: ' + obj.name)
        result['world_aabb_m'] = {'min': _round(min(point[i] for point in world) for i in range(3)),
                                  'max': _round(max(point[i] for point in world) for i in range(3))}
        result['camera_depth_range_m'] = _round([min(depth), max(depth)])
        result['front_vertex_fraction'] = round(sum(value > 0 for value in depth) / len(depth), 7)
        result['all_vertices_in_front'] = min(depth) > 0
        if max(depth) <= 0:
            result['status'] = 'behind_camera'
            return result
        near, far = camera.data.clip_start, camera.data.clip_end
        if max(depth) < near or min(depth) > far:
            result['status'] = 'outside_camera_depth'
            return result
        # Project the actual evaluated triangles. Clipping their depth first
        # avoids mirrored behind-camera boxes and unbounded divisions at z=0.
        mesh.calc_loop_triangles()
        low, high = [math.inf, math.inf], [-math.inf, -math.inf]
        intersects = False
        for triangle in mesh.loop_triangles:
            polygon = [tuple(camera_points[index]) for index in triangle.vertices]
            polygon = _clip_polygon(polygon, 2, -near, False)
            polygon = _clip_polygon(polygon, 2, -far, True)
            if not polygon:
                continue
            image_points = []
            for x, y, z in polygon:
                # Matrix-vector multiplication without creating persistent
                # scene objects; Blender/OpenGL uses local -Z camera forward.
                vector = (x, y, z, 1)
                clip = [sum(projection[row][col] * vector[col] for col in range(4)) for row in range(4)]
                if abs(clip[3]) <= 1e-14:
                    continue
                point = ((clip[0] / clip[3] + 1) / 2, (1 - clip[1] / clip[3]) / 2)
                image_points.append(point)
                for axis in range(2):
                    low[axis], high[axis] = min(low[axis], point[axis]), max(high[axis], point[axis])
            if not intersects and image_points:
                clipped = image_points
                for axis, limit, greater in ((0, 0, True), (0, 1, False), (1, 0, True), (1, 1, False)):
                    clipped = _clip_polygon(clipped, axis, limit, greater)
                intersects = _area(clipped) > 1e-14
        if not all(math.isfinite(value) for value in low + high):
            result['status'] = 'outside_camera_depth'
            return result
        result['projected_bbox_normalized'] = _round(low + high)
        clipped_low = [max(0, low[i]) for i in range(2)]
        clipped_high = [min(1, high[i]) for i in range(2)]
        projected_area = math.prod(max(0, high[i] - low[i]) for i in range(2))
        clipped_area = math.prod(max(0, clipped_high[i] - clipped_low[i]) for i in range(2))
        if clipped_area > 0:
            result['frame_clipped_bbox_normalized'] = _round(clipped_low + clipped_high)
        result['projected_bbox_in_frame_fraction'] = round(min(1, clipped_area / projected_area), 7) if projected_area else 0
        result['frame_bbox_coverage_fraction'] = round(clipped_area, 7)
        result['intersects_frame_before_occlusion'] = bool(intersects)
        fully = (min(depth) >= near and max(depth) <= far
                 and all(low[i] >= -1e-7 and high[i] <= 1 + 1e-7 for i in range(2)))
        result['status'] = ('in_frame' if fully else 'partially_in_frame') if intersects else 'outside_frame'
        if not result['render_enabled']:
            result['status'] = 'hidden_from_render'
        return result
    finally:
        evaluated.to_mesh_clear()


def custom_part_projections(scene, groups, roots, view_id=''):
    """One report per active render camera, with no scene mutation or ray cast."""
    camera = scene.camera
    if camera is None:
        raise ValueError('Custom part projection requires the active render camera')
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(depsgraph,
        x=scene.render.resolution_x, y=scene.render.resolution_y,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y)
    records = []
    if len(groups) != len(roots):
        raise ValueError('Custom part projection roots do not match declared groups')
    for group, root in zip(groups, roots):
        by_index = {obj.get('custom_part_index'): obj for obj in root.children_recursive
                    if obj.get('custom_group_id') == group['id'] and obj.get('custom_part_index') is not None}
        for index, part in enumerate(group['parts']):
            if index not in by_index:
                raise ValueError('Custom part projection missing stable part identity: ' + group['id'] + '/' + str(index))
            obj = by_index[index]
            records.append({'part_id': group['id'] + '/' + str(index), 'group_id': group['id'],
                            'part_index': index, 'part_type': part['type'], 'object_name': obj.name,
                            **_project_part(obj, camera, projection, depsgraph)})
    counts = {}
    for record in records:
        counts[record['status']] = counts.get(record['status'], 0) + 1
    return {'schema_version': 1, 'view_id': view_id, 'camera': camera.name,
            'resolution_px': [scene.render.resolution_x, scene.render.resolution_y],
            'box_convention': '[left,top,right,bottom] normalized image coordinates; origin top-left; unclipped boxes may exceed 0..1',
            'scope': 'Evaluated geometry projected and depth/frame clipped; no occlusion, transparency, material/lighting contrast or visual fidelity test. Bbox fractions are rectangle overlap, not visible surface area.',
            'occlusion_tested': False, 'part_count': len(records), 'status_counts': counts, 'parts': records}
