# Window appearance independent of illumination

Optional saved V2 field, required empty list on new API wire schema:

```json
"exterior_contexts": [{"window_id":"window_main","texture_id":"outside_building",
                       "depth_m":0.2,"brightness_ev":1.0}]
```

`window_id` must refer to one expanded base instance with category `window`;
only one context per instance. `texture_id` references a rectified crop in
`textures` / `texture_sources`; the quadrilateral area must be <= 0.5 of its
source photo, not a whole-room image. The model must select actual exterior
content within the window, excluding the frame and interior. `depth_m` is
0.2..20 metres outward along the window's local +Y. `brightness_ev` is -3..6,
and affects visible exterior pixels only, not interior diffuse illumination.

The renderer makes one fixed photo-textured plane behind the shared window,
with emission used only for camera/glossy/transmission paths. Shadow/diffuse
rays see transparent; it neither blocks daylight nor lights interior objects.
The window must be near an exterior room boundary, with local +Y outward;
invalid attachment rejects preflight. The common plane size covers the aperture
from all supplied cameras, bounded to at most 12 times the aperture dimensions.
This is recorded as a 2.5D distant exterior approximation with limited parallax,
not reconstructed exterior geometry. It never replaces a room surface and
does not change per camera. Actual daylight remains explicit AREA/world lights.

New module `exterior_context.py`; report `exterior_context.json` plus preflight,
annotations and provenance. Source schema lives in `spec_v2.py`.

Validated on Blender 5.2 Cycles: diffuse receiver radiance without/with a +6 EV
context was 1.89904225 / 1.89908254 (ratio 1.0000212); disabling physical daylight
left < 1e-6 diffuse receiver radiance. A camera aimed through clear glazing
changed from black to 0.146113 mean linear radiance with a +1 EV exterior crop.
Cycles emission sampling is disabled (`emission_sampling=NONE`) as well as ray
visibility/shader guards; otherwise it can perturb light sampling variance.
Glossy reflected exterior is intentionally visible and is not diffuse lighting.
