from pcbm.data.metashift import load_metashift_data
train_loader, test_loader, idx_to_class = load_metashift_data(None, scenario=1, seed=42, batch_size=32, num_workers=4)
class_to_idx = {v:k for k,v in idx_to_class.items()}
classes = list(class_to_idx.keys())

print(classes)
print(train_loader.dataset[0], test_loader.dataset[0])