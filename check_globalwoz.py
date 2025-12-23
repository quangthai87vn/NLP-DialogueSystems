import seacrowd as sc

try:
    dset = sc.load_dataset("globalwoz", schema="seacrowd")
    print("Loaded OK. Splits:", dset.keys())
except Exception as e:
    print("LOAD FAILED (expected if you haven't placed local data yet).")
    print("---- error ----")
    print(e)
