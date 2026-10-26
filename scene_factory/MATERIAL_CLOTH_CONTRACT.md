# Additive material and curtain contract

Existing V2/V3 JSON remains valid, without rewriting historical files.

Integration note for the coordinator: `PART_SCHEMA` and `V2_SCHEMA` remain
strict API wire schemas (all properties required), as required by model_api.
`cloth` is an array (`[]` for non-curtains; one parameter record for curtains).
The local `_walk` validator accepts omission of the three new fields only,
without mutation, so prior reviews/edits/specs retain exact scene identity.
`model_api._walk_string_constraints` currently directly indexes every schema
property and needs `if key in value` before recursing to work with this legacy
compatibility. Staged native details must expose `cloth`, copy it into expanded
parts, and staged plans must expose/copy `base_material_overrides`.

`shared_scene.base_material_overrides` is an optional list (maximum 256):

```json
[{"object_id":"desk_main","material_id":"mat_desk_wood","selector":{"role":"body","material_name":"","slot_index":-1}}]
```

`object_id` is an **expanded base instance ID** (same IDs as support references;
repeated base entries require `id__i_j`, not the unexpanded template ID).
`material_id` references `shared_scene.materials`. Creating a material alone
does not bind it. `selector` is optional; omitting it selects all renderable
material slots in this instance. When provided, its three fields are required.
`role` is `all`, `body`, `metal`, `glass`, or `fabric`; role and nonempty name and
nonnegative slot index are intersected. `body` excludes metal, glass and fabric.
Roles are inferred from source shaders/material names and recorded in the
binding report. Prefer `source_material_key` (stable across Blender numeric
name suffixes) or `source_material_id` from that report for `material_name`.
Exact current material names are also accepted. `slot_index=-1` matches every
used slot. Unused mesh slots cannot satisfy a binding. Original source slots are used
for all selections, so order does not change selector meaning. Zero matches,
unknown IDs or conflicting overlapping bindings reject preflight. Overrides
use Blender OBJECT slots so repeated instances sharing a mesh stay independent.
The report records requested bindings, matched slots, actual assignments and
available source slots; creating a material is never reported as an assignment.

A custom part can use `type="curtain"`. Keep standard part fields, with empty
`points`, `vertices`, `faces`, `radius=0`, and add singleton array `cloth`:

```json
[{"fold_count":8,"fold_depth":0.10,"sag":0.025,"thickness":0.002,
 "segments_x":64,"segments_z":24,"seed":23}]
```

`size=[width,depth_envelope,height]` bounds the entire local cloth, including
thickness. Folds run vertically in local XZ; front is -Y. Position is its centre,
then part and group rotations apply as for other parts. `fold_depth` is the
peak-to-peak fold depth in metres, `sag` is vertical edge displacement in metres.
Use `segments_x >= fold_count*4`, `segments_z >= 4`; max 128 by 64. Depth and
sag must fit the envelope. A seeded mesh, continuous textile UVs, smooth normals
and finite thickness are generated; no simulation or executable model code.
Generated vertices and faces count against the existing total geometry budget.
`cloth=[]` for all other primitive types. Strict API schemas require the new
`base_material_overrides`, `selector`, and `cloth` fields. Runtime validation
accepts their omission for saved older JSON without mutating it or its hash.
Use this instead of flat box/low-resolution mesh approximations when a real
hanging textile is visible. This is a bounded parametric cloth approximation,
not a guarantee of exact folds or arbitrary draping.
