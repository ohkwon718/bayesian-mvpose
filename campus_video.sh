python tracking.py --json-dataset ./dataset_settings/epfl_campus.json --path-out ./results --visualize
python eval.py --dataset campus --dir ./results/epfl_campus --re --smoothing --sigma 5
