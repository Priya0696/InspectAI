from inspectai.ingestion.pipeline import ingest_images

records = ingest_images(
    input_dir="data/raw/mvtec/bottle",
    output_file="data/manifests/bottle_manifest.csv",
)

total = len(records)
valid = sum(record["valid"] for record in records)
duplicates = sum(record["is_duplicate"] for record in records)
invalid = total - valid

print("=== InspectAI: MVTec Bottle Ingestion ===")
print(f"Total image files: {total}")
print(f"Valid images: {valid}")
print(f"Invalid images: {invalid}")
print(f"Exact duplicates: {duplicates}")
print("Manifest saved: data/manifests/bottle_manifest.csv")