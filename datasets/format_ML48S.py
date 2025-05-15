import json
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pp = argparse.ArgumentParser(description='Format L48 metadata.')
pp.add_argument('--load-path', type=str, default='./data/v2', help='Path to a directory containing a copy of the L48 dataset.')
pp.add_argument('--save-path', type=str, default='./data/v2', help='Path to output directory.')
pp.add_argument('--save-boxes', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Save box information')
pp.add_argument('--seed', type=int, default=0, help='np random seed.')
args = pp.parse_args()

np.random.seed(args.seed)

assets = pd.read_json(f'{args.load_path}/assets.json')
clips = pd.read_json(f'{args.load_path}/clips.json')
taxa = pd.read_csv(f'{args.load_path}/taxa.csv')
fpaths = clips['file_path'].values
paths = {'train': [], 'val': [], 'test': []}
labels = {'train': [], 'val': [], 'test': []}
annotated_mask = {'train': [], 'val': [], 'test': []}
targ_mask = {'train': [], 'val': [], 'test': []}
target_and_checklist_mask = {'train': [], 'val': [], 'test': []}
target_and_geo_mask = {'train': [], 'val': [], 'test': []}
bboxes = {split:{'image_name': [], 'bbox': [], 'bbox_norm': [], 'cat_idx': []} for split in ['train', 'val', 'test']}

# initialize metadata dictionary:
meta = {}
meta['category_id_to_index'] = {}
meta['category_list'] = []
for i, cat in taxa.iterrows():
    meta['category_list'].append(cat['species_code'])
    meta['category_id_to_index'][i] = i

val_split = np.load(f'{args.load_path}/val_split.npy')

for i in range(len(fpaths)):
    split = assets[assets['id'] == clips.iloc[i]['asset_id']]['split'].values[0]
    if clips.iloc[i]['asset_id'] in val_split:
        split = 'val'
    targ = assets[assets['id'] == clips.iloc[i]['asset_id']]['target_species_code'].values[0]

    labs = np.zeros(100)
    annotated = np.ones(100)
    t_mask = np.zeros(100)
    t_and_checklist_mask = np.ones(100)
    t_and_geo_mask = np.ones(100)

    for cat in clips.iloc[i]['present_species_codes']:
        labs[np.where(taxa['species_code'] == cat)] = 1
    for cat in clips.iloc[i]['unknown_species_codes']:
        annotated[np.where(taxa['species_code'] == cat)] = 0
    for cat in assets[assets['id'] == clips.iloc[i]['asset_id']]['observed_species_codes'].values[0]:
        t_and_checklist_mask[np.where(taxa['species_code'] == cat)] = 0 # any species on checklist could be possible (annotation not reliable)
    for cat in assets[assets['id'] == clips.iloc[i]['asset_id']]['possible_species_codes'].values[0]:
        t_and_geo_mask[np.where(taxa['species_code'] == cat)] = 0 # any species on checklist could be possible (annotation not reliable)

    assert (labs[t_and_geo_mask != 0] == 0).all() and (labs[t_and_checklist_mask != 0] == 0).all() # All confirmed stuff should be negative, ruled based on geo/checklist alone

    t_mask[np.where(taxa['species_code'] == targ)] = 1 if targ in clips.iloc[i]['present_species_codes'] else 0
    t_and_checklist_mask[np.where(taxa['species_code'] == targ)] = 1 if targ in clips.iloc[i]['present_species_codes'] else 0 # Target species prediction is accurate
    t_and_geo_mask[np.where(taxa['species_code'] == targ)] = 1 if targ in clips.iloc[i]['present_species_codes'] else 0 # Target species prediction is accurate

    if args.save_boxes:
        for box in clips.iloc[i]['boxes']:
            bboxes[split]['image_name'].append(clips.iloc[i]['file_path'])
            bboxes[split]['bbox'].append([box['bbox'][1] * clips.iloc[i]['width'], box['bbox'][0] * clips.iloc[i]['height'], box['bbox'][3] * clips.iloc[i]['width'], box['bbox'][2] * clips.iloc[i]['height']])
            bboxes[split]['cat_idx'].append(int(np.where(taxa['species_code'] == box['species_code'])[0]))
            bboxes[split]['bbox_norm'].append([box['bbox'][1], box['bbox'][0], box['bbox'][3], box['bbox'][2]])

    paths[split].append(clips.iloc[i]['file_path'])
    labels[split].append(labs)
    annotated_mask[split].append(annotated)
    targ_mask[split].append(t_mask)
    target_and_checklist_mask[split].append(t_and_checklist_mask)
    target_and_geo_mask[split].append(t_and_geo_mask)


for split in ['train', 'val', 'test']:
    annotated_mask[split] = np.stack(annotated_mask[split])
    targ_mask[split] = np.stack(targ_mask[split])
    target_and_checklist_mask[split] = np.stack(target_and_checklist_mask[split])
    target_and_geo_mask[split] = np.stack(target_and_geo_mask[split])
    labels[split] = np.stack(labels[split])
    paths[split] = np.array(paths[split])

    np.save(os.path.join(args.save_path, 'full_' + split + '_mask.npy'), annotated_mask[split])
    np.save(os.path.join(args.save_path, 'target_' + split + '_mask.npy'), targ_mask[split])
    np.save(os.path.join(args.save_path, split + '_labels.npy'), labels[split])
    np.save(os.path.join(args.save_path, split + '_images.npy'), paths[split])
    np.save(os.path.join(args.save_path, 'checklist_' + split + '_mask.npy'), target_and_checklist_mask[split])
    np.save(os.path.join(args.save_path, 'geo_' + split + '_mask.npy'), target_and_geo_mask[split])
    if args.save_boxes:
        with open(os.path.join(args.save_path, split + '_formatted_bbox.json'), 'w') as f:
            json.dump(bboxes[split], f)

# save metadata: 
with open(os.path.join(args.save_path, 'formatted_metadata.json'), 'w') as f:
    json.dump(meta, f)
