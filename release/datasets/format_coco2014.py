import json
import os
import argparse
import numpy as np
from PIL import Image

pp = argparse.ArgumentParser(description='Format COCO metadata.')
pp.add_argument('--load-path', type=str, default='data/coco2014', help='Path to a directory containing a copy of the COCO dataset.')
pp.add_argument('--img-load-path', type=str, default='data/coco2014', help='Path to a directory containing a copy of the COCO dataset.')
pp.add_argument('--save-path', type=str, default='data/coco2014', help='Path to output directory.')
pp.add_argument('--seed', type=int, default=0, help='np random seed.')
args = pp.parse_args()

np.random.seed(args.seed)

def parse_categories(categories):
    category_list = []
    id_to_index = {}
    for i in range(len(categories)):
        category_list.append(categories[i]['name'])
        id_to_index[categories[i]['id']] = i
    return (category_list, id_to_index)

# initialize metadata dictionary:
meta = {}
meta['category_id_to_index'] = {}
meta['category_list'] = []

for split in ['train', 'val']:
    
    with open(os.path.join(args.load_path, 'annotations', 'instances_' + split + '2014.json'), 'r') as f:
        D = json.load(f)
    
    if len(meta['category_list']) == 0:
        # parse the category data:
        (meta['category_list'], meta['category_id_to_index']) = parse_categories(D['categories'])
    else:
        # check that category lists are consistent for train2014 and val2014:
        (category_list, id_to_index) = parse_categories(D['categories'])
        assert category_list == meta['category_list']
        assert id_to_index == meta['category_id_to_index']

    image_id_list = sorted(np.unique([str(D['annotations'][i]['image_id']) for i in range(len(D['annotations']))]))
    image_id_list = np.array(image_id_list, dtype=int)
    # sorting as strings for backwards compatibility 
    image_id_to_index = {image_id_list[i]: i for i in range(len(image_id_list))}
    
    num_categories = len(D['categories'])
    num_images = len(image_id_list)
    
    label_matrix = np.zeros((num_images,num_categories))
    image_ids = np.zeros(num_images)
    spml = np.zeros((num_images,num_categories))
    box_areas = np.zeros(num_images)

    bbox = {'image_name': [], 'bbox': [], 'bbox_norm': [], 'cat_idx': []}
    
    for i in range(len(D['annotations'])):
        
        image_id = int(D['annotations'][i]['image_id'])
        row_index = image_id_to_index[image_id]
    
        category_id = int(D['annotations'][i]['category_id'])
        category_index = int(meta['category_id_to_index'][category_id])
        
        label_matrix[row_index][category_index] = 1
        image_ids[row_index] = int(image_id)

        if D['annotations'][i]['area'] > box_areas[row_index]:
            spml[row_index, :] = 0
            spml[row_index, category_index] = 1
            box_areas[row_index] = D['annotations'][i]['area']

        bbox['image_name'].append('{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(image_id)).zfill(12)))
        bbox['cat_idx'].append(category_index)
        bb = D['annotations'][i]['bbox']
        bbox['bbox'].append([bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3]])
        with Image.open(os.path.join(args.img_load_path, bbox['image_name'][-1])) as I_raw:
            im = I_raw.convert('RGB')
        bbox['bbox_norm'].append([bbox['bbox'][-1][0] / im.width, bbox['bbox'][-1][1] / im.height, bbox['bbox'][-1][2] / im.width, bbox['bbox'][-1][3] / im.height])
    
    image_ids = np.array(['{}2014/COCO_{}2014_{}.jpg'.format(split, split, str(int(x)).zfill(12)) for x in image_ids])
    
    # save labels and corresponding image ids:
    np.save(os.path.join(args.save_path, 'full_' + split + '_mask.npy'), np.ones(label_matrix.shape))
    np.save(os.path.join(args.save_path, 'largest_box_' + split + '_mask.npy'), spml)
    np.save(os.path.join(args.save_path, '' + split + '_labels.npy'), label_matrix)
    np.save(os.path.join(args.save_path, '' + split + '_images.npy'), image_ids)
    with open(os.path.join(args.save_path, split + '_formatted_bbox.json'), 'w') as f:
        json.dump(bbox, f)

# save metadata: 
with open(os.path.join(args.save_path, 'formatted_metadata.json'), 'w') as f:
    json.dump(meta, f)
