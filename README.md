# ML48S Benchmarking Code

[Data](https://msid-ml48s.s3.amazonaws.com/v0/ml48s.tar.gz)

## Training
### Dataset Processing
First download the ML48S data and the [COCO2014](https://cocodataset.org/#download) data. Run `datasets/format_ML48S.py` and `datasets/format_COCO.py`. 
This should create masks for the fully-labeled, target-only, geo, and checklist versions of the data, as well as image labels and image paths.

### Training
Run `train.py` and input the desired arguments into the argparser. For instance:

`train.py --lr 0.00001 --dataset COCO2014 --train-set-variant geo_seed_1200 --mlab-loss-func EML --eml-alpha 0.1`

The list of methods and their corresponding arguments are given in `losses/get_loss.py`.

To enable asset regularization, run with `--asset-reg` and specify the parameters `--asset-reg-alpha, asset-reg-eps`.

## Dataset Organization (Modified from Appendix A.2)

We organize the ML48S by images in sets which come from recordings, which we also call clips (`clips.json`) and assets (`assets.json`), respectively. 

### Image Directories
Each asset has its own directory where each clip is enumerated sequentially in time. The overall structure of the images is `images/[asset_id]/[clip_num].jpg`.

### Asset Metadata

Each asset has associated metadata which we summarize in the table below and also explain in detail below.

| Field                    | Possible Values   | Description                                       |
| ------------------------ | ----------------- | ------------------------------------------------- |
| `id`                     | \[0, 9999]        | The unique ID associated with the asset           |
| `split`                  | train, test       | Denotes training split or test split for an asset |
| `target_species_code`    | 6-letter-code     | The target species for this asset                 |
| `possible_species_codes` | \[6-letter-codes] | A list of possible species based on ranges        |
| `observed_species_codes` | \[6-letter-codes] | A list of species in the affiliated checklist     |
| `present_species_codes`  | \[6-letter-codes] | A list of positively labeled species              |
| `unknown_species_codes`  | \[6-letter-codes] | All species not in present or absent lists        |
| `absent_species_codes`   | \[6-letter-codes] | A list of negatively labeled species              |

Each asset is associated with a unique asset ID from 0 to 9999. Assets with an ID greater than or equal to 8000 are test assets, and each species has 80 training assets and 20 test assets. For our experiments, we randomly selected 10 training assets per species to serve as validation assets for hyperparameter tuning. Each asset contains a variable number of clips, with a minimum of 11 and a maximum of 1450. As discussed in the paper, every asset has a target species which is provided in the form of a 6-character target species code. The corresponding taxonomic information such as phylogeny, common name, and scientific name are given in `taxa.csv`.

Assets also contain compiled lists of positives, negatives, and unknowns, where positives are also known as present species and negatives are also known as absent species. The list of positives is the union of positives given across each clip in the asset, while the list of negatives is the intersection of clip negatives. The list of unknowns is the species which are not in either of the previous two lists.

Assets also contain two additional fields, possible species given by geographic priors and observed species within the associated checklist. Using the location and time of year each recording was taken, we are able to generate a list of possible species based on species ranges. Though this list does not provide positive labels, absence of a species on this list implies a negative label for that species across the entire recording. This logic also applies for observed species within the associated checklist. Any species present in the recording should also be reported in the associated checklist, so species not on the checklist should have negative labels for the recording. The negative labels generated through checklist data is a superset of the negative labels generated from geographical priors. Hence, geographical priors and checklist data provide two additional levels of weak supervision which falls between SPML and full-labels. We apply negative labels from geographical and range priors to the clip level, even for unlabeled data.

### Clip Metadata

| Field                   | Possible Values   | Description                                            |
| ----------------------- | ----------------- | ------------------------------------------------------ |
| `id`                    | \[0, 416534]      | The unique ID associated with the clip                 |
| `asset_id`              | \[0, 9999]        | The asset ID from which this clip came                 |
| `clip_order`            | \[0, 1449]        | The position of the clip within the asset              |
| `file_path`             | Relative filepath | The path to the image for the given clip               |
| `width`                 | 750               | The image width                                        |
| `height`                | 236               | The image height                                       |
| `present_species_codes` | \[6-letter-codes] | A list species with positive labels                    |
| `unknown_species_codes` | \[6-letter-codes] | All species not in present or absent lists             |
| `absent_species_codes`  | \[6-letter-codes] | A list species with negative labels                    |
| `boxes`                 | \[dictionaries]   | Bounding box annotations for the clip (see next table) |

Clips also have corresponding metadata which is summarized in the table above. The bounding box annotations for each clip are provided, where each box is specified with an ID, species code, status, and coordinates. The box ID is unique to a clip, so no two boxes within the same clip share the same ID. The bounding box coordinates are given in relative coordinates falling within \[0, 1] and are provided as `[xmin, ymin, xmax, ymax]`. For box status, sounds which are longer than 80 ms which are only present in the first or last 200 ms of a window are labeled `"ignore"` while others are `"active"`.

Boxes which do not have status `"ignore"` are treated as positive labels for the multi-label task and are given in the list of positives. Any clip with positive labels is treated as fully-labeled, meaning all other species are negative, unless there are `"Unknown bird"` boxes, in which case we treat other possible species as unknown (but retain negatives from geographical priors).
Box metadata is given in detail below:

| Field          | Possible Values                     | Description                                    |
| -------------- | ----------------------------------- | ---------------------------------------------- |
| `id`           | int                                 | Box ID unique to each clip                     |
| `species_code` | 6-letter-code                       | The species which this vocalization belongs to |
| `status`       | `"passive"`, `"active"`, `"ignore"` | Species prevalence in the clip                 |
| `bbox`         | `[0, 1]^4`                          | Box coordinates `[xmin, ymin, xmax, ymax]`     |

