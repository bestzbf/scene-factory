"""Real GLB normalization, native PBR and support-patch tests; no API calls."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    import bpy
except ImportError:
    bpy = None
from asset_intake import atomic_json,sha,_record_for
from asset_intake_blender import compile_asset,inspect_glb,support_surfaces
from asset_catalog import AssetCatalog
if bpy is not None:
    from external_assets import ExternalAssetImporter

@unittest.skipUnless(bpy is not None, 'Requires Blender Python')
class IntakeBlenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        bpy.ops.wm.read_factory_settings(use_empty=True)
    def fixture(self,textured=False):
        source=self.root/'source';source.mkdir()
        bpy.ops.mesh.primitive_cube_add(size=2,location=(2,3,4));obj=bpy.context.object;obj.scale=(1,.5,.5)
        material=bpy.data.materials.new('Paint');material.use_nodes=True;obj.data.materials.append(material)
        bsdf=material.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value=(.2,.4,.6,1)
        bsdf.inputs['Roughness'].default_value=.23;bsdf.inputs['Metallic'].default_value=.7
        if textured:
            nodes,links=material.node_tree.nodes,material.node_tree.links
            image=bpy.data.images.new('PackedMR',width=2,height=2);image.colorspace_settings.name='Non-Color'
            image.pixels[:]=[1,.25,.8,1]*4;image.pack();tex=nodes.new('ShaderNodeTexImage');tex.image=image
            separate=nodes.new('ShaderNodeSeparateColor');links.new(tex.outputs['Color'],separate.inputs[0])
            links.new(separate.outputs['Green'],bsdf.inputs['Roughness']);links.new(separate.outputs['Blue'],bsdf.inputs['Metallic'])
            normal=bpy.data.images.new('Normal',width=2,height=2);normal.colorspace_settings.name='Non-Color';normal.pixels[:]=[.5,.5,1,1]*4;normal.pack()
            texture=nodes.new('ShaderNodeTexImage');texture.image=normal;mapping=nodes.new('ShaderNodeNormalMap');mapping.inputs['Strength'].default_value=.42
            links.new(texture.outputs['Color'],mapping.inputs['Color']);links.new(mapping.outputs['Normal'],bsdf.inputs['Normal'])
        bpy.ops.export_scene.gltf(filepath=str(source/'model.glb'),export_format='GLB',use_selection=True)
        metadata={'schema_version':'scene_asset_input_v1','category':'table','placement':'floor','longest_dimension_m':2,'scale_source':'estimated'}
        atomic_json(source/'asset.json',metadata)
        return source,metadata
    def compiled(self,textured=False):
        source,metadata=self.fixture(textured);output=self.root/'out';output.mkdir()
        compile_asset(source/'model.glb',source/'asset.json',output,'table_01')
        receipt=json.loads((output/'worker_receipt.json').read_text());atomic_json(output/'receipt.json',receipt)
        asset,material=_record_for('table_01',metadata,receipt,'model.glb','receipt.json',sha(output/'receipt.json'))
        catalog=AssetCatalog({'schema_version':'asset_catalog_v1','root':str(output),'assets':[asset],
                              'material_variants':{material['material_variant_id']:material}})
        return output,receipt,catalog
    def test_scale_origin_surfaces_and_native_scalar_material_isolation(self):
        output,receipt,catalog=self.compiled()
        self.assertEqual(receipt['status'],'technical_ready');self.assertFalse(receipt['visual_accepted'])
        dims=receipt['asset']['dimensions_import_units']
        for a,b in zip(dims,[2,1,1]):self.assertAlmostEqual(a,b,places=5)
        surfaces=receipt['asset']['support_surfaces'];self.assertTrue(surfaces)
        self.assertAlmostEqual(surfaces[-1]['z'],1,places=5)
        importer=ExternalAssetImporter(catalog)
        roots=[]
        for identifier in ('one','two'):
            roots.append(importer.instantiate({'id':identifier,'category':'table','position':[0,0,0],'rotation_degrees':[0,0,0]},
                {'object_id':identifier,'asset_id':'table_01','material_variant_id':'table_01__native','uniform_scale':1}))
        mats=[next(o for o in root.children if o.type=='MESH').data.materials[0] for root in roots]
        self.assertIsNot(mats[0],mats[1])
        for mat in mats:
            bsdf=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
            self.assertAlmostEqual(bsdf.inputs['Roughness'].default_value,.23,places=5)
            self.assertAlmostEqual(bsdf.inputs['Metallic'].default_value,.7,places=5)
            for a,b in zip(bsdf.inputs['Base Color'].default_value,[.2,.4,.6,1]):self.assertAlmostEqual(a,b,places=5)
        mats[0].node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value=.9
        self.assertAlmostEqual(mats[1].node_tree.nodes.get('Principled BSDF').inputs['Roughness'].default_value,.23,places=5)
    def test_packed_mr_and_normal_survive_roundtrip(self):
        output,receipt,catalog=self.compiled(textured=True)
        document=inspect_glb(output/'model.glb');material=document['materials'][0]
        self.assertIn('metallicRoughnessTexture',material['pbrMetallicRoughness'])
        self.assertIn('normalTexture',material);self.assertAlmostEqual(material['normalTexture']['scale'],.42,places=5)
        self.assertTrue(all('bufferView' in image for image in document['images']))
        self.assertTrue(catalog.production_readiness('table_01')[0])
    def test_scene_builder_without_overrides_preserves_native_appearance(self):
        from test_multiview import fixture
        from generate_multiview import build_scene
        output,receipt,catalog=self.compiled()
        bundle=fixture();spec=bundle['shared_scene']
        spec['textures']=[];spec['custom_objects']=[];spec['exterior_contexts']=[]
        spec['base_material_overrides']=[]
        spec['base_scene']['objects']=[{'id':'native_table','category':'table','position':[0,2,0],
            'dimensions':[2,1,1],'rotation_z_deg':0,'color':[.9,.1,.1],
            'repeat':[1,1],'spacing':[0,0],'confidence':1}]
        handles=build_scene(bundle,asset_catalog=catalog,asset_bindings={'schema_version':1,'instances':[
            {'object_id':'native_table','asset_id':'table_01','material_variant_id':'table_01__native','uniform_scale':1}]})
        self.assertEqual(handles['material_binding']['applied_slot_count'],0)
        root=handles['roots'][0]
        mat=next(o for o in root.children if o.type=='MESH').data.materials[0]
        shader=next(n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
        self.assertAlmostEqual(shader.inputs['Roughness'].default_value,.23,places=5)
        for a,b in zip(shader.inputs['Base Color'].default_value,[.2,.4,.6,1]):self.assertAlmostEqual(a,b,places=5)
    def test_nonopaque_glb_rejected(self):
        from asset_intake_blender import inspect_glb
        source,_=self.fixture()
        mat=bpy.data.materials.get('Paint');mat.node_tree.nodes.get('Principled BSDF').inputs['Alpha'].default_value=.4
        bpy.ops.export_scene.gltf(filepath=str(source/'alpha.glb'),export_format='GLB',use_selection=True)
        with self.assertRaisesRegex(ValueError,'opaque'):inspect_glb(source/'alpha.glb')

    def mesh_object(self, vertices, faces, name='Patch'):
        mesh=bpy.data.meshes.new(name+'Mesh');mesh.from_pydata(vertices,[],faces);mesh.update()
        obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
        material=bpy.data.materials.new(name+'Material');material.use_nodes=True;obj.data.materials.append(material)
        return obj

    def test_near_planar_noise_is_admitted_without_bbox_top(self):
        vertices=[];faces=[];nx,ny=12,8
        for iy in range(ny+1):
            for ix in range(nx+1):
                x=-1+2*ix/nx;y=-.6+1.2*iy/ny;z=1+.00018*math.sin(ix*1.7+iy*.9)
                vertices.append((x,y,z))
        for iy in range(ny):
            for ix in range(nx):
                a=iy*(nx+1)+ix;b=a+1;c=a+nx+1;d=c+1
                faces.extend(((a,b,d),(a,d,c)))
        obj=self.mesh_object(vertices,faces)
        surfaces=support_surfaces([obj])
        self.assertTrue(surfaces)
        self.assertLess(surfaces[0]['area_m2'],2.4)
        self.assertLess(max(p[1] for p in surfaces[0]['polygon_xy'])-min(p[1] for p in surfaces[0]['polygon_xy']),1.2)

    def test_hole_is_not_filled_by_convex_hull(self):
        # A broad grid with a central 0.4 m square hole. The admitted
        # rectangle may use one side, but it must never cross the opening.
        vertices=[];faces=[];nx=20;ny=12
        for iy in range(ny+1):
            for ix in range(nx+1):vertices.append((-1+2*ix/nx,-.6+1.2*iy/ny,1.0))
        for iy in range(ny):
            for ix in range(nx):
                x=-1+2*(ix+.5)/nx;y=-.6+1.2*(iy+.5)/ny
                if -.2<x<.2 and -.2<y<.2:continue
                a=iy*(nx+1)+ix;b=a+1;c=a+nx+1;d=c+1;faces.extend(((a,b,d),(a,d,c)))
        obj=self.mesh_object(vertices,faces)
        surfaces=support_surfaces([obj])
        self.assertTrue(surfaces)
        for surface in surfaces:
            poly=surface['polygon_xy'];xs=[p[0] for p in poly];ys=[p[1] for p in poly]
            self.assertFalse(min(xs)<-.2<max(xs) and min(ys)<.2 and max(ys)>-.2)

    def test_tilted_patch_is_rejected(self):
        vertices=[];faces=[];nx=12;ny=8
        for iy in range(ny+1):
            for ix in range(nx+1):
                x=-1+2*ix/nx;y=-.6+1.2*iy/ny;vertices.append((x,y,1+.08*x))
        for iy in range(ny):
            for ix in range(nx):
                a=iy*(nx+1)+ix;b=a+1;c=a+nx+1;d=c+1;faces.extend(((a,b,d),(a,d,c)))
        self.assertFalse(support_surfaces([self.mesh_object(vertices,faces)]))

    def test_occluding_face_is_excluded_from_admitted_patch(self):
        base=self.mesh_object([(-1,-.6,1),(1,-.6,1),(1,.6,1),(-1,.6,1)],[(0,1,2),(0,2,3)],'Base')
        bpy.ops.mesh.primitive_cube_add(size=1,location=(0,0,1.5));block=bpy.context.object
        surfaces=support_surfaces([base,block])
        self.assertTrue(surfaces)
        for surface in surfaces:
            if surface['z']>1.1:continue # The occluder's own exposed top is valid.
            poly=surface['polygon_xy'];xs=[p[0] for p in poly];ys=[p[1] for p in poly]
            self.assertFalse(min(xs)<.5 and max(xs)>-.5 and min(ys)<.5 and max(ys)>-.5)

    def test_curved_patch_is_not_reported_as_broad_flat_support(self):
        vertices=[];faces=[];n=32
        for iy in range(n+1):
            for ix in range(n+1):
                x=-1+2*ix/n;y=-1+2*iy/n
                vertices.append((x,y,1+.2*(x*x+y*y)))
        for iy in range(n):
            for ix in range(n):
                a=iy*(n+1)+ix;b=a+1;c=a+n+1;d=c+1;faces.extend(((a,b,d),(a,d,c)))
        self.assertFalse(support_surfaces([self.mesh_object(vertices,faces)]))

    def test_union_coverage_rejects_tiny_hole_even_with_duplicate_overlap(self):
        from asset_intake_blender import _polygons_cover_rect
        # Area sums are deliberately misleading: duplicate left coverage
        # exceeds the narrow missing strip's area but must not fill it.
        left=[(0,0),(.49999,0),(.49999,1),(0,1)]
        right=[(.50001,0),(1,0),(1,1),(.50001,1)]
        self.assertFalse(_polygons_cover_rect([left,left,right],0,1,0,1))
        a=[(0,0),(1,0),(1,1)];b=[(0,0),(1,1),(0,1)]
        self.assertTrue(_polygons_cover_rect([a,a,b],0,1,0,1))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntakeBlenderTests))
    if not result.wasSuccessful():raise SystemExit(1)
