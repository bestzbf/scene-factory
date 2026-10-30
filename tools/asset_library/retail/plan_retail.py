"""Plan the retail asset set and check it against the pilot's requested sizes.

Writes specs/<asset_id>.json (one build spec per asset) and plan_report.json.
The coverage check uses the scene factory's own size-fit rule (one uniform scale
in 0.8..1.25, worst-axis relative error), so "covered" means the selector would
accept the asset for that request. Pilot category names are aliased to the
planned library categories only for this estimate; the real run lets the model
read the new categories from the library hints.

    python plan_retail.py [--work-dir WORK_DIR] [--write]
                          [--base-catalog CATALOG.json --runs RUN_DIR [RUN_DIR ...]]

--write writes WORK_DIR/specs/*.json (one per asset) for build_all.py. The plan itself
needs no input. The optional coverage report needs the base library catalog and one or
more scene-factory output directories whose scenes hold initial_layout.json.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path.cwd()   # replaced by --work-dir when run as a script
PILOTS = []          # --runs
BASE_CATALOG = None  # --base-catalog
TOLERANCE, SCALE = 0.25, (0.8, 1.25)

ALIASES = {
    'commercial_glass_door_refrigerator': 'glass_door_refrigerator', 'display_refrigerator': 'glass_door_refrigerator',
    'glass_top_chest_freezer': 'chest_freezer', 'glass_lid_chest_freezer': 'chest_freezer',
    'retail_gondola_shelf': 'shelf', 'retail_display_rack': 'wire_display_rack',
    'large_water_jug': 'water_jug', 'packaged_water_case': 'bottled_water_pack', 'bottled_water_multipack': 'bottled_water_pack',
    'retail_counter': 'checkout_counter', 'display_counter': 'glass_display_counter', 'refrigerated_display_counter': 'glass_display_counter',
    'point_of_sale_terminal': 'pos_terminal', 'cash_register': 'pos_terminal',
    'countertop_retail_display': 'countertop_display_rack', 'countertop_merchandise_display': 'countertop_display_rack',
    'desktop_monitor': 'computer_monitor', 'pos_monitor': 'computer_monitor',
    'waste_bin': 'trash_bin', 'pedal_bin': 'trash_bin', 'retail_product_box': 'cardboard_box',
    'countertop_food_steamer': 'heated_food_display', 'glass_display_case': 'glass_display_counter',
}


def log_grid(low, high, step):
    values, v = [], low
    while v < high * 1.0001:
        values.append(round(v, 3)); v *= step
    if values[-1] < high * 0.97:
        values.append(round(high, 3))
    return values


# category -> list of families: (archetype, placement, params variants, W grid, D grid, H grid, filter)
FAMILIES = {
    'shelf': [
        ('wall_shelf', 'floor', [{'frame': 'white_metal'}, {'frame': 'white_metal', 'header': True},
                                 {'frame': 'grey_metal'}, {'frame': 'light_wood'}],
         log_grid(0.45, 5.0, 1.3), [0.3, 0.4, 0.52, 0.66], [1.3, 1.65, 2.05, 2.45], None),
        ('gondola', 'floor', [{'frame': 'white_metal'}, {'frame': 'grey_metal'}, {'frame': 'white_metal', 'header': True}],
         log_grid(0.6, 5.0, 1.3), [0.75, 1.0, 1.35, 1.8], [1.3, 1.65, 2.05], None),
        ('gondola', 'floor', [{'frame': 'white_metal', 'orient': 'y'}, {'frame': 'grey_metal', 'orient': 'y'}],
         [0.4, 0.55, 0.75, 1.0], log_grid(0.9, 5.0, 1.35), [1.3, 1.65, 2.05], lambda w, d, h: d > 1.3 * w),
    ],
    'glass_door_refrigerator': [
        ('glass_door_cooler', 'floor', [{'body': 'black_plastic', 'canopy': 'red_accent'}, {'body': 'black_plastic', 'canopy': 'black_plastic'},
                                        {'body': 'white_metal', 'canopy': 'blue_accent'}, {'body': 'grey_metal', 'canopy': 'white_metal'}],
         log_grid(0.6, 3.6, 1.25), [0.62, 0.78, 0.95], [1.95, 2.2], None),
    ],
    'open_display_refrigerator': [
        ('open_cooler', 'floor', [{'body': 'white_metal', 'accent': 'red_accent'}, {'body': 'black_plastic', 'accent': 'blue_accent'}],
         log_grid(1.0, 4.0, 1.3), [0.75, 0.95], [1.9, 2.15], None),
    ],
    'chest_freezer': [
        ('chest_freezer', 'floor', [{'body': 'white_metal'}, {'body': 'white_metal', 'accent': 'blue_accent'},
                                    {'body': 'white_metal', 'accent': 'red_accent'}],
         log_grid(0.6, 2.6, 1.25), [0.58, 0.72, 0.9, 1.15], [0.85], None),
        ('chest_freezer', 'floor', [{'body': 'white_metal', 'orient': 'y'}],
         [0.6, 0.75, 0.95], log_grid(1.1, 2.6, 1.3), [0.85], lambda w, d, h: d > 1.3 * w),
    ],
    'checkout_counter': [
        ('checkout_counter', 'floor', [{'body': 'white_plastic', 'front_panel': 'red_accent'}, {'body': 'light_wood'},
                                       {'body': 'white_plastic', 'display_front': True}, {'body': 'grey_plastic', 'front_panel': 'blue_accent'}],
         log_grid(0.6, 4.5, 1.25), [0.5, 0.65, 0.85, 1.1], [0.88], None),
        ('checkout_counter', 'floor', [{'body': 'white_plastic', 'orient': 'y'}, {'body': 'light_wood', 'orient': 'y'}],
         [0.5, 0.65, 0.85], log_grid(1.0, 4.0, 1.25), [0.88], lambda w, d, h: d > 1.3 * w),
    ],
    'cabinet': [
        ('storage_cabinet', 'floor', [{'body': 'light_wood'}, {'body': 'white_plastic'}, {'body': 'wood'}],
         log_grid(0.6, 4.0, 1.3), [0.42, 0.58], [0.85, 1.05], None),
        ('tobacco_cabinet', 'floor', [{'body': 'light_wood'}, {'body': 'wood'}, {'body': 'white_metal'}],
         log_grid(0.7, 4.0, 1.3), [0.38, 0.5, 0.65], [1.55, 1.95, 2.4], None),
    ],
    'glass_display_counter': [
        ('display_counter', 'floor', [{'body': 'light_wood'}, {'body': 'white_metal'}, {'body': 'wood'}],
         log_grid(0.8, 3.0, 1.3), [0.5, 0.65], [0.95, 1.1], None),
        ('display_counter', 'floor', [{'body': 'light_wood', 'orient': 'y'}],
         [0.55, 0.7], log_grid(1.2, 3.0, 1.3), [0.95, 1.1], lambda w, d, h: d > 1.3 * w),
    ],
    'wire_display_rack': [
        ('wire_rack', 'floor', [{}, {'fill': 0.7}], [0.45, 0.6, 0.8], [0.4, 0.6, 0.85], [1.0, 1.35, 1.7], None),
        ('wire_rack', 'floor', [{'orient': 'y'}], [0.45, 0.55], [1.0, 1.4], [1.0, 1.35, 1.7], lambda w, d, h: d > 1.3 * w),
    ],
    'countertop_display_rack': [
        ('countertop_display', 'surface', [{'body': 'chrome'}, {'body': 'white_plastic'}],
         [0.3, 0.45, 0.65, 0.95, 1.4, 1.8], [0.22, 0.32, 0.5, 0.75], [0.12, 0.25, 0.45, 0.7, 1.1], None),
    ],
    'produce_display': [
        ('produce_display', 'floor', [{'body': 'wood'}, {'body': 'light_wood'}], [0.75, 1.0, 1.4, 1.9], [0.6, 0.8], [0.8, 0.95], None),
        ('produce_display', 'floor', [{'body': 'wood', 'orient': 'y'}], [0.7, 0.9], [1.5, 2.1], [0.85], lambda w, d, h: d > 1.3 * w),
    ],
    'cardboard_box': [
        ('carton', 'floor', [{'plain': True}, {'plain': False}],
         log_grid(0.25, 0.75, 1.25), log_grid(0.22, 0.95, 1.3), log_grid(0.16, 0.6, 1.3), None),
    ],
    'parcel': [('carton', 'floor', [{'plain': True}], [0.3, 0.45], [0.25, 0.36], [0.1, 0.16], None)],
    'bottled_water_pack': [
        ('bottle_multipack', 'floor', [{'cols': 4, 'rows': 3}, {'cols': 3, 'rows': 2}, {'cols': 4, 'rows': 6}],
         [0.3, 0.4, 0.5], [0.26, 0.36, 0.5], [0.22, 0.31], None),
    ],
    'water_jug': [('water_jug', 'floor', [{}], [0.26, 0.32], [0.26, 0.32], [0.4, 0.5], lambda w, d, h: abs(w - d) < 0.01)],
    'plastic_crate': [
        ('plastic_crate', 'floor', [{'colour': 'blue_accent'}, {'colour': 'red_accent', 'stocked': True}, {'colour': 'green_accent'},
                                    {'colour': 'yellow_accent', 'stocked': True}],
         [0.4, 0.52, 0.68], [0.33, 0.45, 0.6], [0.22, 0.32, 0.45, 0.6], None),
    ],
    'bottle': [('bottle', 'surface', [{'cap': 'red_accent'}, {'cap': 'blue_accent'}, {'cap': 'green_accent', 'glass': 'green_accent'}],
                [0.065, 0.085, 0.105], None, [0.2, 0.26, 0.32], None)],
    'tin_can': [('can', 'surface', [{}], [0.055, 0.07], None, [0.1, 0.13, 0.17], None)],
    'instant_noodle_cup': [('cup_noodle', 'surface', [{}], [0.1, 0.13, 0.16], None, [0.1, 0.14, 0.18], None)],
    'snack_bag': [('snack_bag', 'surface', [{}], [0.16, 0.22, 0.3], [0.08, 0.14, 0.2], [0.22, 0.3, 0.38], None)],
    'pos_terminal': [('pos_terminal', 'surface', [{'drawer': True}, {'drawer': False}], [0.32, 0.45, 0.62], [0.28, 0.4, 0.55],
                      [0.3, 0.42, 0.52], None)],
    'payment_terminal': [('payment_terminal', 'surface', [{}], [0.08, 0.12, 0.2, 0.3], [0.1, 0.15, 0.22], [0.1, 0.18, 0.3], None)],
    'barcode_scanner': [('barcode_scanner', 'surface', [{}, {'window': 'green_accent'}], [0.08, 0.12, 0.15], [0.1, 0.14, 0.18],
                         [0.07, 0.12, 0.19], None)],
    'computer_monitor': [('monitor', 'surface', [{}], [0.36, 0.48, 0.6], [0.16, 0.22], [0.32, 0.4, 0.48], None)],
    'basket': [('basket', 'surface', [{'colour': 'red_accent'}, {'colour': 'blue_accent'}, {'colour': 'green_accent'}],
                [0.25, 0.34, 0.45], [0.19, 0.27, 0.38], [0.12, 0.18, 0.24], None)],
    'shopping_basket_stack': [('basket_stack', 'floor', [{'colour': 'red_accent'}, {'colour': 'blue_accent'}],
                               [0.42, 0.5], [0.32, 0.38], [0.4, 0.55, 0.72], None)],
    'fire_extinguisher': [('fire_extinguisher', 'floor', [{}], [0.15, 0.19], None, [0.42, 0.55, 0.65], None)],
    'trash_bin': [('bin', 'floor', [{'colour': 'grey_plastic'}, {'colour': 'white_plastic', 'lid': 'grey_plastic'}],
                   [0.22, 0.28, 0.35], None, [0.28, 0.42, 0.6], None)],
    'water_dispenser': [('water_dispenser', 'floor', [{}], [0.3, 0.38], [0.32, 0.4], [1.0, 1.25], None)],
    'self_service_kiosk': [('kiosk', 'floor', [{}], [0.42, 0.55], [0.42, 0.55], [1.4, 1.6], lambda w, d, h: abs(w - d) < 0.01)],
    'weighing_scale': [('scale', 'surface', [{}], [0.3, 0.42], [0.3, 0.42], [0.1, 0.14], None)],
    'printer': [('appliance', 'surface', [{'colour': 'white_plastic', 'window': False}, {'colour': 'black_plastic', 'window': False}],
                 [0.36, 0.46], [0.32, 0.42], [0.2, 0.3], None)],
    'microwave': [('appliance', 'surface', [{'colour': 'white_plastic'}, {'colour': 'grey_metal'}], [0.45, 0.52], [0.34, 0.4],
                   [0.26, 0.31], None)],
    'heated_food_display': [('appliance', 'surface', [{'colour': 'chrome'}, {'colour': 'red_accent'}], [0.38, 0.5, 0.65],
                             [0.36, 0.5], [0.45, 0.58], None)],
    'cash_drawer': [('appliance', 'surface', [{'colour': 'black_plastic', 'window': False}], [0.36, 0.46], [0.36, 0.44],
                     [0.09, 0.12], None)],
    'calculator': [('appliance', 'surface', [{'colour': 'black_plastic', 'window': False}], [0.12, 0.16], [0.16, 0.2],
                    [0.03, 0.045], None)],
    'countertop_sign': [('appliance', 'surface', [{'colour': 'white_plastic', 'window': False}, {'colour': 'red_accent', 'window': False}],
                         [0.12, 0.17, 0.24], [0.05, 0.08], [0.18, 0.26, 0.34], None)],
    'electric_kettle': [('bin', 'surface', [{'colour': 'white_plastic', 'lid': 'black_plastic'}, {'colour': 'chrome', 'lid': 'black_plastic'}],
                         [0.17, 0.21], None, [0.22, 0.28], None)],
    'confectionery_jar': [('bin', 'surface', [{'colour': 'water_plastic', 'lid': 'red_accent'}], [0.13, 0.17, 0.21], None,
                           [0.18, 0.24, 0.3], None)],
    'structural_column': [('appliance', 'floor', [{'colour': 'white_plastic', 'window': False}], [0.35, 0.5, 0.65], None,
                           [2.6, 3.1, 3.6], None)],
    'floor_mat': [('appliance', 'floor', [{'colour': 'grey_plastic', 'window': False}, {'colour': 'red_accent', 'window': False}],
                   [0.45, 0.6, 0.9], [0.6, 0.9, 1.2], [0.012], None)],
    'cardboard_retail_display': [('wire_rack', 'floor', [{}], [0.5, 0.65], [0.32, 0.42], [1.0, 1.2, 1.45], None)],
    'freestanding_stainless_steel_sink': [('sink', 'floor', [{}], [0.5, 0.75, 1.0, 1.3], [0.5, 0.65], [0.85], None)],
    'wash_basin': [('tub', 'surface', [{'colour': 'white_plastic'}, {'colour': 'chrome'}], [0.35, 0.45], [0.28, 0.36], [0.12, 0.18], None)],
    'bucket': [('tub', 'floor', [{'colour': 'blue_accent'}, {'colour': 'red_accent'}], [0.26, 0.32], None, [0.28, 0.36], None)],
    'wire_basket': [('tub', 'floor', [{'colour': 'chrome'}], [0.4, 0.5], [0.4, 0.5], [0.6, 0.75], None)],
    'sack': [('sack', 'floor', [{}], [0.45, 0.6], [0.35, 0.55], [0.55, 0.75], None)],
    'potted_plant': [('plant', 'surface', [{}, {'pot': 'red_accent'}], [0.22, 0.3, 0.4], [0.22, 0.3, 0.4], [0.3, 0.4, 0.55],
                      lambda w, d, h: abs(w - d) < 0.01)],
    'lidded_product_tin': [('bin', 'surface', [{'colour': 'yellow_accent', 'lid': 'red_accent'}], [0.2, 0.3], None, [0.25, 0.38], None)],
    'vacuum_flask': [('flask', 'surface', [{'colour': 'chrome'}, {'colour': 'red_accent'}], [0.12, 0.2, 0.27], None, [0.3, 0.45, 0.6], None)],
    'table_fan': [('fan', 'surface', [{}], [0.25, 0.34], [0.2, 0.26], [0.36, 0.47], None)],
    'mop': [('mop', 'floor', [{}], [0.3], [0.15], [1.2, 1.35], None)],
    'tissue_box': [('carton', 'surface', [{'plain': False}], [0.2, 0.24], [0.11, 0.17], [0.09, 0.13], None)],
    'keyboard': [('appliance', 'surface', [{'colour': 'black_plastic', 'window': False}], [0.36, 0.44], [0.13, 0.16], [0.025, 0.035], None)],
    'tablet_with_stand': [('monitor', 'surface', [{}], [0.22, 0.27], [0.14, 0.18], [0.22, 0.27], None)],
    'tabletop_mirror': [('monitor', 'surface', [{}], [0.2, 0.26], [0.1, 0.13], [0.26, 0.32], None)],
    'pen_holder': [('bin', 'surface', [{'colour': 'black_plastic', 'lid': 'black_plastic'}], [0.08, 0.1], None, [0.1, 0.15], None)],
    'cooler': [('cooler_box', 'floor', [{'body': 'blue_accent'}, {'body': 'red_accent'}], [0.4, 0.5, 0.65], [0.35, 0.48], [0.38, 0.45], None)],
    'tobacco_display': [('countertop_display', 'surface', [{'body': 'light_wood'}], [1.2, 1.8, 2.6], [0.3, 0.4], [0.9, 1.25], None)],
    'desk': [('desk', 'floor', [{'top': 'light_wood'}, {'top': 'white_plastic'}], [1.0, 1.25, 1.5], [0.5, 0.65, 0.8], [0.75, 0.82], None)],
    'seat_cushion': [('cushion', 'surface', [{}], [0.4, 0.48], [0.4, 0.48], [0.04, 0.08], None)],
    'throw_pillow': [('cushion', 'surface', [{'colour': 'red_accent'}], [0.32, 0.4], [0.3, 0.38], [0.08, 0.12], None)],
}

# Stackable goods appear both on the floor and on other items (cartons on cartons,
# water on water, baskets on counters). One asset declares one placement, so these
# get a twin family with the same geometry and the other placement. New families
# are appended, so existing asset IDs stay stable.
TWINS = {'cardboard_box': 'surface', 'plastic_crate': 'surface', 'bottled_water_pack': 'surface', 'parcel': 'surface',
         'basket': 'floor', 'water_jug': 'surface'}
for _category, _placement in TWINS.items():
    FAMILIES[_category] = FAMILIES[_category] + [(arch, _placement, variants, ws, ds, hs, keep)
                                                 for arch, _, variants, ws, ds, hs, keep in FAMILIES[_category]]
# Furniture the model may load with items gets a twin whose top (or steps) is left
# clear: the stocked version looks right on its own, the clear one leaves room for
# what the model saw on it. Appended families keep earlier asset IDs stable.
for _category in ('shelf', 'countertop_display_rack'):
    FAMILIES[_category] = FAMILIES[_category] + [(arch, placement, [{**v, 'top_stock': False} for v in variants], ws, ds, hs, keep)
                                                 for arch, placement, variants, ws, ds, hs, keep in FAMILIES[_category]]
FAMILIES['computer'] = [('monitor', 'surface', [{}], [0.45, 0.58, 0.72], [0.25, 0.38, 0.5], [0.4, 0.5], None)]

# Second real run (new library hints): categories the model asked for that the library lacked,
# and sizes outside the uniform-scale range. Size families are appended so earlier IDs stay stable.
NO_WINDOW = {'window': False}
FAMILIES.update({
    'claw_machine': [('appliance', 'floor', [{'colour': 'red_accent'}, {'colour': 'blue_accent'}],
                      [0.6, 0.8, 1.0], [0.55, 0.8], [0.8, 1.7, 1.95], None),
                     ('appliance', 'surface', [{'colour': 'red_accent'}, {'colour': 'blue_accent'}],
                      [0.45, 0.6], [0.4, 0.55], [0.6, 0.8], None)],
    'power_bank_rental_station': [('appliance', 'surface', [{'colour': 'white_plastic'}, {'colour': 'blue_accent'}],
                                   [0.25, 0.32, 0.4], [0.2, 0.26], [0.3, 0.45, 0.6], None)],
    'paper_roll': [('appliance', 'surface', [{'colour': 'white_plastic', **NO_WINDOW}], [0.2, 0.25, 0.3], [0.06, 0.08, 0.1],
                    [0.06, 0.08, 0.1], None)],
    'food_steamer': [('appliance', 'surface', [{'colour': 'chrome'}, {'colour': 'white_plastic'}], [0.36, 0.45, 0.55],
                      [0.36, 0.45], [0.45, 0.55], None)],
    'computer_mouse': [('appliance', 'surface', [{'colour': 'black_plastic', **NO_WINDOW}], [0.06, 0.07], [0.1, 0.12],
                        [0.035, 0.045], None)],
    'refrigerator': [('appliance', 'floor', [{'colour': 'white_plastic', **NO_WINDOW}, {'colour': 'grey_metal', **NO_WINDOW}],
                      [0.55, 0.65, 0.75], [0.55, 0.65], [1.45, 1.85, 2.0], None)],
    'cooking_oil_jug': [('bottle', 'surface', [{'cap': 'red_accent', 'glass': 'yellow_accent'}, {'cap': 'green_accent', 'glass': 'yellow_accent'}],
                         [0.15, 0.19, 0.23], None, [0.28, 0.35, 0.42], None)],
    'retail_gift_box': [('carton', 'surface', [{'plain': False}], [0.2, 0.25, 0.32, 0.4], [0.15, 0.2, 0.28],
                         [0.2, 0.28, 0.36, 0.45], None)],
    'packaged_tissue': [('carton', 'surface', [{'plain': False}], [0.3, 0.4, 0.5], [0.2, 0.25, 0.3], [0.12, 0.17, 0.22], None)],
    'packaged_disposable_cups': [('can', 'surface', [{}], [0.08, 0.1, 0.13], None, [0.2, 0.28, 0.36], None)],
    'shopping_bag_bundle': [('sack', 'surface', [{}], [0.35, 0.45], [0.25, 0.33], [0.18, 0.24, 0.32], None)],
    'insulated_delivery_bag': [('cooler_box', 'surface', [{'body': 'yellow_accent'}, {'body': 'blue_accent'}],
                                [0.3, 0.38, 0.45], [0.28, 0.35], [0.25, 0.3, 0.38], None)],
    'cleaning_product_container': [('appliance', 'surface', [{'colour': 'white_plastic', **NO_WINDOW}, {'colour': 'blue_accent', **NO_WINDOW}],
                                    [0.12, 0.19], [0.09, 0.13], [0.18, 0.23, 0.3], None)],
    'metal_collection_box': [('appliance', 'floor', [{'colour': 'grey_metal', **NO_WINDOW}], [0.3, 0.4], [0.25, 0.32],
                              [0.4, 0.5, 0.65], None)],
    'stool': [('desk', 'floor', [{'top': 'light_wood'}, {'top': 'white_plastic'}], [0.3, 0.4], [0.13, 0.2, 0.3, 0.4], [0.42, 0.5], None)],
    'towel': [('cushion', 'surface', [{}], [0.3, 0.4], [0.25, 0.35], [0.05, 0.09], None)],
    'storage_bin': [('tub', 'floor', [{'colour': 'black_plastic'}, {'colour': 'blue_accent'}], [0.4, 0.5], [0.35, 0.45], [0.35, 0.45], None)],
})
FAMILIES['wire_display_rack'].append(('wire_rack', 'floor', [{}], [1.05, 1.4], [0.4, 0.6], [1.35, 1.7], None))
FAMILIES['structural_column'].append(('appliance', 'floor', [{'colour': 'white_plastic', **NO_WINDOW}], [0.15, 0.2, 0.25], None,
                                      [2.6, 3.1, 3.6], None))
FAMILIES['payment_terminal'].append(('payment_terminal', 'surface', [{}], [0.08, 0.12], [0.1, 0.15], [0.055], None))
FAMILIES['printer'].append(('appliance', 'surface', [{'colour': 'white_plastic', **NO_WINDOW}, {'colour': 'black_plastic', **NO_WINDOW}],
                            [0.2, 0.28], [0.22, 0.3], [0.15, 0.2], None))
# Small items go on the floor in one shop and on a counter or shelf in the next. One asset
# declares one placement, so every small family not already twinned gets the other one.
SMALL = 0.8
for _category, _families in list(FAMILIES.items()):
    if _category in TWINS:
        continue
    places = {f[1] for f in _families}
    extra = []
    for arch, placement, variants, ws, ds, hs, keep in _families:
        other = 'floor' if placement == 'surface' else 'surface'
        if other not in places and max(ws + (ds or []) + hs) <= SMALL:
            extra.append((arch, other, variants, ws, ds, hs, keep))
    FAMILIES[_category] = _families + extra

FLOOR_ONLY_SQUARE = {'tin_can', 'bottle', 'instant_noodle_cup', 'fire_extinguisher', 'trash_bin', 'electric_kettle', 'confectionery_jar',
                     'structural_column', 'water_jug'}


def plan():
    specs = []
    for category, families in FAMILIES.items():
        for f_index, (archetype, placement, variants, ws, ds, hs, keep) in enumerate(families):
            ds = ds or [None]
            combos = []
            for w, d, h in itertools.product(ws, ds, hs):
                d = w if d is None else d                 # round items: depth = width
                if keep and not keep(w, d, h):
                    continue
                combos.append((w, d, h))
            for i, (w, d, h) in enumerate(combos):
                params = dict(variants[i % len(variants)])
                orient = params.pop('orient', None)
                asset_id = f"retail_{category}_{f_index}_{i:03d}"
                specs.append({'asset_id': asset_id, 'category': category, 'archetype': archetype,
                              'size_m': [w, d, h], 'placement': placement,
                              'seed': int.from_bytes(asset_id.encode()[-6:], 'little') % 100000,
                              'params': {**params, **({'orient': orient} if orient else {})}})
    return specs


def fit(source, target):
    ratios = [a / b for a, b in zip(source, target)]
    scale = min(SCALE[1], max(SCALE[0], 2 / (min(ratios) + max(ratios))))
    return max(abs(a * scale - b) / b for a, b in zip(source, target))


def coverage(specs):
    base = json.loads(BASE_CATALOG.read_text())
    sizes = {}
    # Keyed by (category, placement): the selector only offers floor assets for floor requests
    # and table/surface assets for objects resting on another object.
    for a in base['assets']:
        place = 'floor' if a['placement'] == 'floor' else 'surface'
        sizes.setdefault((a['category'], place), []).append(('base', [v * a['meters_per_unit'] for v in a['dimensions_import_units']]))
    for s in specs:
        sizes.setdefault((s['category'], s['placement']), []).append(('retail', s['size_m']))
    rows = {}
    for scene in sorted(p for pilot in PILOTS for p in pilot.iterdir() if (p / 'initial_layout.json').is_file()):
        for o in json.loads((scene / 'initial_layout.json').read_text())['objects']:
            c = ALIASES.get(o['category'], o['category'])
            place = 'floor' if o['support_id'] == 'floor' else 'surface'
            best = min(((fit(sz, o['target_dimensions_m']), src) for src, sz in sizes.get((c, place), [])), default=(None, None))
            rows.setdefault(c, []).append(best)
    report = {}
    for c, values in sorted(rows.items(), key=lambda kv: -len(kv[1])):
        errors = [e for e, _ in values if e is not None]
        report[c] = {'requests': len(values), 'in_library': any(key[0] == c for key in sizes),
                     'covered_025': sum(e <= TOLERANCE for e in errors), 'covered_015': sum(e <= 0.15 for e in errors),
                     'mean_best_error': round(sum(errors) / len(errors), 3) if errors else None,
                     'won_by_retail': sum(src == 'retail' for e, src in values if e is not None and e <= TOLERANCE)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-catalog', type=Path)
    parser.add_argument('--runs', nargs='+', type=Path, default=[])
    parser.add_argument('--work-dir', type=Path, default=Path.cwd())
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    BASE_CATALOG = args.base_catalog.resolve() if args.base_catalog else None
    PILOTS, ROOT = [p.resolve() for p in args.runs], args.work_dir.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    specs = plan()
    out = ROOT / 'specs'
    if args.write:
        out.mkdir(exist_ok=True)
        for old in out.glob('*.json'):
            old.unlink()
        for s in specs:
            (out / f"{s['asset_id']}.json").write_text(json.dumps(s) + '\n')
    by_category = {}
    for s in specs:
        by_category[s['category']] = by_category.get(s['category'], 0) + 1
    print(f'planned assets: {len(specs)} in {len(by_category)} categories')
    report = {}
    if BASE_CATALOG and PILOTS:
        report = coverage(specs)
        total = sum(r['requests'] for r in report.values())
        covered = sum(r['covered_025'] for r in report.values())
        print(f'pilot objects covered (tol 0.25, aliased names): {covered}/{total} = {covered / max(total, 1):.0%}')
        for c, r in report.items():
            flag = '' if r['covered_025'] == r['requests'] else '  <-- gap'
            print(f"  {c:<34} {r['covered_025']:>3}/{r['requests']:<3} (<=0.15: {r['covered_015']:>3}) mean err {r['mean_best_error']}{flag}")
    (ROOT / 'plan_report.json').write_text(json.dumps({'assets': len(specs), 'by_category': by_category,
                                                      'pilot_coverage': report}, indent=1) + '\n')
