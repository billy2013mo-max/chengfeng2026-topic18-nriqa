import json, csv

t150_data = json.load(open('pred_nriqa_dataset_test.json'))
with open('data/manifest_test_150.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name'])
    for x in t150_data:
        writer.writerow([x['image']])

print('Created data/manifest_test_150.csv with', len(t150_data), 'images.')
