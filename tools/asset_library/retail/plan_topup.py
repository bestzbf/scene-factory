"""Turn a list of unmet object requests into asset specs at exactly the requested size.

    python plan_topup.py REQUESTS.json --work-dir WORK_DIR [--write] [--min-scenes 1]

REQUESTS.json is a list of {"category", "placement", "size_m": [L, W, H], "objects", "scenes"}
built from scene-factory runs: every object a layout asked for that the library could not
supply. Building at the exact requested size means the selector's uniform-scale rule
(0.8..1.25, <=25% per-axis error) always accepts the result, so the only question left is
whether the shape reads as the right thing.

Each category maps to one of the procedural archetypes in build_retail_asset.py. Where no
archetype fits the object's shape, MAP picks the nearest generic one rather than inventing
geometry that will fail the builder's own size check (it rejects a build whose measured
bounds differ from the spec by more than 0.75..1.34). Archetypes that fill their bounding
box are preferred: appliance, carton, sack, cushion, cooler_box, desk.

Writes WORK_DIR/specs/*.json for build_all.py, plus WORK_DIR/topup_report.json.
"""
import argparse
import json
from pathlib import Path

# category -> (archetype, extra params). Rounded things use bin/can/flask/water_jug (cylinders);
# boxy things use appliance/carton (full bounding box); furniture uses the matching builder.
MAP = {
    # --- tall floor appliances
    'pedestal_fan': ('fan', {}), 'floor_fan': ('fan', {}), 'tower_fan': ('fan', {}),
    'floor_standing_air_conditioner': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'air_conditioner': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'portable_air_conditioner': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'portable_electric_heater': ('appliance', {'colour': 'grey_metal', 'window': False}),
    'vending_machine': ('kiosk', {}), 'parcel_locker': ('kiosk', {}), 'parcel_locker_bank': ('kiosk', {}),
    'photocopier': ('kiosk', {}), 'office_copier': ('kiosk', {}),
    'refrigerator': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'undercounter_refrigerator': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'glass_display_refrigerator': ('glass_door_cooler', {}), 'small_glass_display_refrigerator': ('glass_door_cooler', {}),
    'combination_display_freezer': ('glass_door_cooler', {}),
    'commercial_steamer': ('appliance', {'colour': 'chrome'}), 'bain_marie': ('appliance', {'colour': 'chrome'}),
    'food_warming_cabinet': ('appliance', {'colour': 'chrome'}),
    # --- counters, tables, cabinets
    'checkout_counter': ('checkout_counter', {'body': 'white_plastic'}),
    'refrigerated_work_counter': ('checkout_counter', {'body': 'white_metal'}),
    'refrigerated_preparation_counter': ('checkout_counter', {'body': 'white_metal'}),
    'stainless_preparation_table': ('desk', {'top': 'chrome'}),
    'retail_display_table': ('desk', {'top': 'light_wood'}), 'folded_table': ('desk', {'top': 'white_plastic'}),
    'console_table': ('desk', {'top': 'wood'}),
    'cabinet': ('storage_cabinet', {'body': 'light_wood'}), 'banknote_counter': ('checkout_counter', {'body': 'grey_plastic'}),
    'cash_register': ('pos_terminal', {'drawer': True}), 'money_counting_machine': ('pos_terminal', {'drawer': False}),
    'tabletop_service_terminal': ('pos_terminal', {'drawer': False}), 'touchscreen_terminal': ('pos_terminal', {'drawer': False}),
    'medicine_bag_sealing_machine': ('appliance', {'colour': 'grey_metal', 'window': False}),
    'electronic_weighing_scale': ('scale', {}), 'retail_weighing_scale': ('scale', {}),
    'countertop_scale': ('scale', {}), 'weighing_scale': ('scale', {}),
    'office_copier': ('kiosk', {}), 'computer_tower': ('appliance', {'colour': 'black_plastic', 'window': False}),
    'laptop': ('appliance', {'colour': 'grey_metal', 'window': True}),
    'telephone': ('appliance', {'colour': 'black_plastic', 'window': False}),
    'small_speaker': ('appliance', {'colour': 'black_plastic', 'window': False}), 'speaker': ('appliance', {'colour': 'black_plastic', 'window': False}),
    'payment_speaker': ('appliance', {'colour': 'black_plastic', 'window': False}),
    'surveillance_camera': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'electronic_camera_device': ('appliance', {'colour': 'grey_metal', 'window': False}),
    'toaster_oven': ('appliance', {'colour': 'grey_metal'}), 'countertop_oven': ('appliance', {'colour': 'grey_metal'}),
    'rice_cooker': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'coffee_machine': ('appliance', {'colour': 'black_plastic'}), 'commercial_beverage_dispenser': ('water_dispenser', {}),
    'insulated_beverage_dispenser': ('water_dispenser', {}), 'vacuum_pump_dispenser': ('water_dispenser', {}),
    'hot_water_dispenser': ('water_dispenser', {}), 'countertop_water_dispenser': ('water_dispenser', {}),
    'soup_warmer': ('appliance', {'colour': 'chrome'}), 'countertop_food_warmer': ('appliance', {'colour': 'chrome'}),
    'hot_dog_roller': ('appliance', {'colour': 'chrome'}), 'hot_dog_roller_grill': ('appliance', {'colour': 'chrome'}),
    'hot_dog_grill': ('appliance', {'colour': 'chrome'}), 'roller_grill': ('appliance', {'colour': 'chrome'}),
    'food_display_cover': ('appliance', {'colour': 'chrome', 'window': False}),
    # --- shelving and racks
    'shelf': ('wall_shelf', {}), 'glass_display_rack': ('wire_rack', {}), 'wire_display_rack': ('wire_rack', {}),
    'plastic_bin_display_rack': ('wire_rack', {}), 'magazine_rack': ('wire_rack', {}),
    'basket_stand': ('wire_rack', {}), 'bakery_display': ('countertop_display', {'body': 'light_wood'}),
    'confectionery_display_island': ('countertop_display', {}), 'bulk_confectionery_display': ('countertop_display', {}),
    'lollipop_display': ('countertop_display', {}), 'tobacco_display': ('countertop_display', {'body': 'light_wood'}),
    'customer_display': ('countertop_display', {}), 'display_column': ('countertop_display', {}),
    'merchandise_platform': ('desk', {'top': 'white_plastic'}),
    'produce_display': ('produce_display', {'body': 'wood'}), 'bulk_bin_display': ('produce_display', {'body': 'wood'}),
    'bulk_food_bin': ('bin', {'colour': 'blue_accent'}), 'bulk_food_bin_counter': ('bin', {'colour': 'white_plastic'}),
    'bulk_food_container': ('bin', {'colour': 'white_plastic'}),
    'produce_tub': ('tub', {'colour': 'blue_accent'}), 'laundry_basket': ('tub', {'colour': 'blue_accent'}),
    'wash_basin_stack': ('tub', {'colour': 'white_plastic'}), 'floor_standing_wash_basin': ('sink', {}),
    # --- sacks, bags and bales
    'sack': ('sack', {}), 'rice_bag': ('sack', {}), 'packaged_rice': ('sack', {}),
    'soft_bag': ('sack', {}), 'shopping_bag': ('sack', {}), 'shopping_bag_bundle': ('sack', {}),
    'gift_bag': ('sack', {}), 'bakery_bag': ('sack', {}), 'trash_bag': ('sack', {}),
    'bedding_bundle': ('sack', {}), 'packaged_bedding': ('sack', {}), 'folded_bedding': ('cushion', {}),
    'wrapped_bundle': ('sack', {}), 'wrapped_long_item': ('cushion', {}),
    'clothes_hanger_bundle': ('sack', {}), 'cable_or_bundle': ('sack', {}),
    'broom': ('mop', {}), 'mop': ('mop', {}),
    'flattened_cardboard_stack': ('carton', {'plain': True}), 'flat_shipping_carton': ('carton', {'plain': True}),
    'cardboard_box': ('carton', {'plain': True}),
    'metal_drum': ('bin', {'colour': 'grey_metal', 'lid': 'dark_metal'}),
    # --- boxes, cartons and packaged goods
    'medicine_box': ('carton', {'plain': False}), 'beverage_carton': ('carton', {'plain': False}),
    'packaged_bread': ('carton', {'plain': False}), 'packaged_noodles': ('carton', {'plain': False}),
    'packaged_product_roll': ('carton', {'plain': False}), 'snack_packet': ('snack_bag', {}),
    'food_packet': ('snack_bag', {}), 'food_tray_package': ('carton', {'plain': False}),
    'ready_meal_tray': ('carton', {'plain': False}), 'lunch_box': ('carton', {'plain': False}),
    'egg_carton': ('carton', {'plain': False}), 'retail_egg_carton': ('carton', {'plain': False}),
    'egg_tray': ('carton', {'plain': False}), 'egg_pack': ('carton', {'plain': False}),
    'retail_gift_box': ('carton', {'plain': False}), 'cigarette_carton': ('carton', {'plain': False}),
    'cigarette_pack': ('carton', {'plain': False}), 'lidded_product_tin': ('bin', {'colour': 'yellow_accent', 'lid': 'red_accent'}),
    'milk_carton_case': ('bottle_multipack', {}), 'beverage_multipack': ('bottle_multipack', {}),
    'beverage_display_tray': ('plastic_crate', {'colour': 'red_accent', 'stocked': True}),
    'dish': ('carton', {'plain': False}), 'plate': ('carton', {'plain': False}),
    # --- cylinders: tubes, rolls, jars, cups
    'snack_tube': ('can', {}), 'ceramic_storage_jar': ('bin', {'colour': 'white_plastic', 'lid': 'light_wood'}),
    'lidded_ceramic_storage_jar': ('bin', {'colour': 'white_plastic', 'lid': 'light_wood'}),
    'condiment_jar': ('bin', {'colour': 'water_plastic', 'lid': 'red_accent'}),
    'paper_roll': ('can', {}), 'drinking_cup': ('cup_noodle', {}), 'paper_cup': ('cup_noodle', {}),
    'bowl': ('tub', {'colour': 'white_plastic'}), 'teacup': ('cup_noodle', {}), 'mug': ('cup_noodle', {}),
    'enamel_lidded_mug': ('cup_noodle', {}), 'utensil_holder': ('bin', {'colour': 'grey_metal', 'lid': 'grey_metal'}),
    'tin_can': ('can', {}), 'bottle': ('bottle', {}),
    # A tray must present a top face: the model puts teapots and cups on it, and only a
    # closed top yields a verified support surface. At H = 4-7 cm the block reads as a slab.
    'tray': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'tea_tray': ('appliance', {'colour': 'light_wood', 'window': False}),
    'food_tray': ('appliance', {'colour': 'chrome', 'window': False}),
    # --- floor soft goods and misc
    'floor_mat': ('cushion', {}), 'rug': ('cushion', {}), 'towel': ('cushion', {}),
    'woven_seat_pad': ('cushion', {}), 'throw_pillow': ('cushion', {}), 'seat_cushion': ('cushion', {}),
    'stool': ('desk', {'top': 'light_wood'}), 'bench': ('desk', {'top': 'light_wood'}),
    'chair': ('desk', {'top': 'blue_accent'}), 'folding_lounge_chair': ('desk', {'top': 'blue_accent'}),
    'reclining_armchair': ('desk', {'top': 'blue_accent'}), 'bed': ('desk', {'top': 'white_plastic'}),
    'potted_plant': ('plant', {}), 'potted_tree': ('plant', {}), 'flower_arrangement': ('plant', {}),
    'structural_column': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'internal_partition': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'glass_partition': ('appliance', {'colour': 'water_plastic', 'window': False}),
    'staircase': ('appliance', {'colour': 'grey_plastic', 'window': False}),
    'laundry_basket': ('tub', {'colour': 'blue_accent'}),
    'hand_truck': ('appliance', {'colour': 'grey_metal', 'window': False}),
    'cart': ('desk', {'top': 'chrome'}),
    'shopping_basket': ('basket', {}), 'wire_basket': ('tub', {'colour': 'chrome'}),
    'basket': ('basket', {}),
    'animal_figurine': ('carton', {'plain': False}), 'sculpture': ('plant', {}), 'trophy': ('flask', {'colour': 'chrome'}),
    'terrestrial_globe': ('flask', {}), 'helmet': ('flask', {}), 'motorcycle_helmet': ('flask', {}),
    'straw_hat': ('cushion', {}), 'umbrella': ('mop', {'colour': 'red_accent'}),
    'slipper_pair': ('cushion', {'colour': 'blue_accent'}),
    'candy_dispenser': ('bin', {'colour': 'water_plastic', 'lid': 'red_accent'}),
    'boxed_toy': ('carton', {'plain': False}), 'fruit': ('flask', {'colour': 'red_accent'}),
    'vegetable': ('flask', {'colour': 'green_accent'}), 'loose_eggs': ('flask', {'colour': 'light_wood'}),
    'egg_batch': ('carton', {'plain': True}),
    'menu': ('cushion', {'colour': 'white_plastic'}), 'framed_certificate': ('cushion', {'colour': 'white_plastic'}),
    'folding_lounge_chair': ('desk', {'top': 'blue_accent'}),
    'freestanding_advertising_sign': ('appliance', {'colour': 'white_plastic', 'window': False}),
    'device_case': ('carton', {'plain': False}), 'pet_food_bag': ('sack', {}),
    'adapter': ('appliance', {'colour': 'black_plastic', 'window': False}),
}
FALLBACK = ('appliance', {'colour': 'grey_plastic', 'window': False})   # fills its bounding box
ROUND = ('bin', {'colour': 'grey_plastic', 'lid': 'grey_plastic'})       # near-round footprint

# Cylindrical archetypes build a round footprint, so the generator's per-axis snap
# (accepted range 0.75..1.34) rejects them for an elongated box. Anything whose footprint
# is wider than that ratio is rebuilt as a box instead, or with the category's own
# rectangular stand-in where one reads better than a plain block.
ROUNDISH = {'bin', 'can', 'flask', 'cup_noodle', 'water_jug', 'bottle', 'plant', 'fire_extinguisher',
            'water_dispenser', 'snack_bag', 'basket', 'trophy'}
RECT_STANDIN = {
    'bulk_food_bin': ('carton', {'plain': True}), 'bulk_food_bin_counter': ('carton', {'plain': True}),
    'bulk_food_container': ('carton', {'plain': True}), 'produce_tub': ('plastic_crate', {'colour': 'blue_accent'}),
    'vegetable': ('carton', {'plain': False}), 'fruit': ('carton', {'plain': False}),
    'loose_eggs': ('carton', {'plain': True}), 'egg_batch': ('carton', {'plain': True}),
    'tray': ('carton', {'plain': True}), 'food_tray': ('carton', {'plain': True}),
    'bulk_confectionery_display': ('plastic_crate', {'colour': 'red_accent', 'stocked': True}),
}
SNAP = 1.34   # build_retail_asset.py rejects a per-axis snap factor outside 0.75..1.34


def slug(category):
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in category)


def plan(requests, min_scenes, sane):
    specs, unmapped = [], set()
    for r in requests:
        cat, size = r['category'], [round(float(v), 3) for v in r['size_m']]
        if r['scenes'] < min_scenes or not sane(size):
            continue
        arch, params = MAP.get(cat, (None, None))
        if arch is None:
            unmapped.add(cat)
            # A round footprint gets a cylinder; anything else a full bounding box.
            arch, params = ROUND if abs(size[0] - size[1]) <= 0.03 * max(size[0], size[1]) and size[2] > 0.5 * min(size[0], size[1]) else FALLBACK
        elif arch in ROUNDISH and max(size[0], size[1]) > SNAP * min(size[0], size[1]):
            # Cylinder geometry cannot stretch to an elongated footprint; rebuild as a box.
            arch, params = RECT_STANDIN.get(cat, FALLBACK)
        # The asset ID carries the archetype: retargeting a category to different geometry
        # (a tray rebuilt as a closed slab so it can carry objects) then produces a new ID
        # instead of an ID reuse that intake correctly refuses.
        specs.append({'asset_id': f"topup_{slug(cat)}_{r['placement'][0]}_{arch}_{len(specs):04d}",
                      'category': cat, 'archetype': arch, 'size_m': size, 'placement': r['placement'],
                      'seed': 1000 + len(specs) * 37 % 90000, 'params': dict(params)})
    return specs, unmapped


def sane(size):
    return all(0.02 <= v <= 6.0 for v in size) and max(size) / max(min(size), 1e-9) <= 12


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('requests', type=Path)
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--min-scenes', type=int, default=1, help='skip requests seen in fewer than N scenes')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    requests = json.loads(args.requests.read_text(encoding='utf-8'))
    specs, unmapped = plan(requests, args.min_scenes, sane)
    root = args.work_dir.resolve()
    if args.write:
        out = root / 'specs'
        out.mkdir(parents=True, exist_ok=True)
        for old in out.glob('*.json'):
            old.unlink()
        for s in specs:
            (out / f"{s['asset_id']}.json").write_text(json.dumps(s) + '\n', encoding='utf-8')
    by_arch, by_cat = {}, {}
    for s in specs:
        by_arch[s['archetype']] = by_arch.get(s['archetype'], 0) + 1
        by_cat[s['category']] = by_cat.get(s['category'], 0) + 1
    print(f'specs: {len(specs)} in {len(by_cat)} categories using {len(by_arch)} archetypes')
    print(f'objects covered: {sum(r["objects"] for r in requests if r["scenes"] >= args.min_scenes and sane(r["size_m"]))}')
    print('archetypes:', dict(sorted(by_arch.items(), key=lambda kv: -kv[1])))
    if unmapped:
        print(f'{len(unmapped)} unmapped categories use a generic shape: {sorted(unmapped)[:40]}')
    (root / 'topup_report.json').write_text(json.dumps(
        {'specs': len(specs), 'by_category': by_cat, 'by_archetype': by_arch,
         'unmapped': sorted(unmapped)}, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
