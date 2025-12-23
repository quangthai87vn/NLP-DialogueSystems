import seacrowd as sc

cfgs = sc.available_config_names("globalwoz")
print("Total configs:", len(cfgs))
for c in cfgs:
    if "_vi_" in c or c.endswith("_vi") or "vietnam" in c.lower():
        print(c)
