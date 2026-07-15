import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import FashionDataValidator

DEEPFASHION_PROCESSED = PROJECT_ROOT / "datasets" / "processed" / "deepfashion_processed.json"
FASHIONGEN_PROCESSED = PROJECT_ROOT / "datasets" / "processed" / "fashiongen_processed.json"
OUTPUT_REPORT = PROJECT_ROOT / "datasets" / "metadata" / "validation_report.json"

def main():
    print("Loading processed sample records...")
    records = []
    
    if DEEPFASHION_PROCESSED.exists():
        with open(DEEPFASHION_PROCESSED, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Some formats save as list, some as dict with a 'records' key
            if isinstance(data, dict) and "records" in data:
                records.extend(data["records"])
            elif isinstance(data, list):
                records.extend(data)
        print(f"Loaded records from DeepFashion: {len(records)}")
    else:
        print(f"DeepFashion processed file not found: {DEEPFASHION_PROCESSED}")

    fashiongen_start_idx = len(records)
    if FASHIONGEN_PROCESSED.exists():
        with open(FASHIONGEN_PROCESSED, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "records" in data:
                f_records = data["records"]
            elif isinstance(data, list):
                f_records = data
            else:
                f_records = []
            
            # Normalize attributes if dict
            for rec in f_records:
                if isinstance(rec.get("attributes"), dict):
                    attr_dict = rec["attributes"]
                    flat_attrs = []
                    for val in attr_dict.values():
                        if isinstance(val, list):
                            flat_attrs.extend(val)
                    rec["attributes"] = flat_attrs
            
            records.extend(f_records)
        print(f"Loaded records from FashionGen: {len(records) - fashiongen_start_idx}")
    else:
        print(f"FashionGen processed file not found: {FASHIONGEN_PROCESSED}")

    if not records:
        print("No records loaded to validate! Please run the ingestion pipelines first.")
        sys.exit(1)

    print(f"Total records to validate: {len(records)}")
    
    # Initialize the validator
    validator = FashionDataValidator(project_root=str(PROJECT_ROOT))
    
    # Run batch validation
    print("Running batch validation...")
    batch_result = validator.validate_batch(records)
    
    # Print validation stats summary
    print("\nBatch Validation Summary:")
    print(f"  Total records   : {batch_result.total_records}")
    print(f"  Valid records   : {batch_result.valid_records}")
    print(f"  Failed records  : {batch_result.failed_records}")
    print(f"  Warning records : {batch_result.warning_records}")
    print(f"  Success rate    : {batch_result.success_rate:.2%}")
    print(f"  Quality score   : {batch_result.quality_score:.4f}")
    
    if batch_result.error_breakdown:
        print("\nError Breakdown:")
        for err, count in batch_result.error_breakdown.items():
            print(f"  - {err}: {count}")

    if batch_result.warning_breakdown:
        print("\nWarning Breakdown:")
        for warn, count in batch_result.warning_breakdown.items():
            print(f"  - {warn}: {count}")

    # Save the report to the metadata directory
    print(f"\nSaving validation report to {OUTPUT_REPORT}...")
    validator.save_report(batch_result, OUTPUT_REPORT)
    print("Report saved successfully!")

if __name__ == "__main__":
    main()
