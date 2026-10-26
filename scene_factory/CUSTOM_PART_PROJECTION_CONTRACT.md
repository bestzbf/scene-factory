# Per-view custom part projection diagnostics

`annotations_<view>.json` and the V2 `annotations.json` gain
`custom_part_projection` (one compact report for the active camera). Every
custom part appears, including behind-camera and completely off-frame parts.
Identity is `part_id="<group_id>/<zero_based_part_index>"`, also supplied as
separate `group_id` and `part_index` fields. This supports all existing 16 views;
changing the camera does not create or move geometry.

Per-part fields: `part_type`, `object_name`, `world_aabb_m` from evaluated mesh,
`camera_depth_range_m`, `front_vertex_fraction` (mesh vertices, not visible area),
`all_vertices_in_front`, `projected_bbox_normalized`,
`frame_clipped_bbox_normalized`, `projected_bbox_in_frame_fraction`,
`frame_bbox_coverage_fraction`, `intersects_frame_before_occlusion`, `status`.
Normalized boxes use `[left,top,right,bottom]`, image origin top-left.
Unclipped boxes may extend outside 0..1 and are null if geometry is fully outside
camera near/far clipping. The in-frame fraction is rectangle overlap / projected
rectangle area, not a silhouette or visibility fraction. Polygons of evaluated
mesh are depth-clipped and frame-clipped to determine frame intersection, so a
large polygon crossing the image still counts even if all vertices are outside.

Statuses: `in_frame`, `partially_in_frame`, `outside_frame`, `behind_camera`,
`outside_camera_depth`, `hidden_from_render`, `missing_geometry`.
This measures geometric projection only. No occlusion, light/material contrast,
alpha/transparent visibility, screen-space relevance or visual correctness is
certified. A reported in-frame curtain can still be hidden by the bed. A curtain
behind camera cannot be claimed as a visible repair.

Implementation: `part_projection.py`, called from `generate_v2.make_annotations`
(therefore every V3 camera automatically receives its own report). New stable
index metadata is recorded by `generate_v2.build_custom` on existing objects;
no diagnostic helper objects are created. Parent/group/part rotation, thickness,
curve bevels and modifiers are evaluated before projection.
