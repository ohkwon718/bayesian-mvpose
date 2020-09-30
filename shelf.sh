python tracking.py --json-dataset ./dataset_settings/Shelf.json --path-out ./results
python eval.py --dataset shelf --dir ./results/Shelf --re --smoothing --sigma 1.5
