import json
import seacrowd as sc

CONFIG = "globalwoz_EandF_vi_seacrowd_tod"   # <-- dán config bạn chọn vào đây

dset = sc.load_dataset_by_config_name(config_name=CONFIG)
print("Splits:", dset.keys())

# lấy 1 sample
split = "test" if "test" in dset else ("validation" if "validation" in dset else "train")
ex = dset[split][0]

print("Example keys:", list(ex.keys()))
print("---- RAW EXAMPLE (first sample) ----")
print(json.dumps(ex, ensure_ascii=False, indent=2)[:2500])  # cắt ngắn cho dễ nhìn
