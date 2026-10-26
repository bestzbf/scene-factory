import copy,json,unittest
from pathlib import Path
from schema import validate
class SceneSchemaTests(unittest.TestCase):
 def setUp(self):
  self.spec=json.loads((Path(__file__).parent/'examples/living_room.json').read_text())
 def test_reject_untrusted_extra_fields(self):
  self.spec['python']='import os'
  with self.assertRaises(ValueError):validate(self.spec)
 def test_reject_excessive_expanded_geometry(self):
  obj=self.spec['objects'][0]
  self.spec['objects']=[{**copy.deepcopy(obj),'id':str(i),'repeat':[12,12]} for i in range(5)]
  with self.assertRaises(ValueError):validate(self.spec)
 def test_reject_invalid_numbers_and_duplicate_ids(self):
  self.spec['room']['height']=float('nan')
  with self.assertRaises(ValueError):validate(self.spec)
  self.spec['room']['height']=3
  self.spec['objects'][1]['id']=self.spec['objects'][0]['id']
  with self.assertRaises(ValueError):validate(self.spec)
 def test_reject_degenerate_camera(self):
  self.spec['camera']['target']=self.spec['camera']['position'][:]
  with self.assertRaises(ValueError):validate(self.spec)
if __name__=='__main__':unittest.main()
