# Locally authored office assets

This asset pack copies the reusable procedural furniture implementation from
`office_reconstruction/furniture.py` and four locally drawn demonstration screen
textures from `office_reconstruction/assets/`. No external model download or
image-generation service is used.

`office_furniture.py` has two compatibility changes: Principled BSDF shaders are
looked up by Blender node type rather than localized display name, and random
seeds are stored as strings to accommodate 64-bit SHA-256-derived values.

`../detailed_assets.py` adapts these assets to the batch schema, uses deterministic
variants, and fits measured mesh bounds to each specified bottom-centre anchor
and dimensions. Extreme dimension ratios will distort assets; they should be
flagged by scene-spec quality checks. Monitor textures are required dependencies
and are packed into generated `.blend` files.
