import pandas as pd
from pathlib import Path

input_dir = Path("/home/realtmxi/Github/muss/sinhala/MADLAD_CulturaX_cleaned/data/parquet")
output_dir = Path("/home/realtmxi/Github/muss/sinhala/MADLAD_CulturaX_cleaned/data/")

# Convert each parquet file to a text file
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

for parquet_file in input_dir.glob("*.parquet"):
    print(f"Processing {parquet_file.name}...")  # Debug: Show which file is being processed
    try:
        df = pd.read_parquet(parquet_file)
        
        # Ensure the 'text' column exists
        if 'text' not in df.columns:
            raise ValueError(f"'text' column not found in {parquet_file.name}. Check the dataset schema.")
        
        # Save sentences to a text file
        output_file = output_dir / f"{parquet_file.stem}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in df['text']:
                if isinstance(line, str):
                    f.write(line.strip() + "\n")
        print(f"Converted {parquet_file.name} to {output_file.name}")
    except Exception as e:
        print(f"Error processing {parquet_file.name}: {e}")

print("Conversion completed.")