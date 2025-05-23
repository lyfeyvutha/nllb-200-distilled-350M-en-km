import csv
import json
import os
from typing import Dict, Any
from tqdm import tqdm

def convert_csv_to_nllb_format(
    csv_file_path: str, 
    output_file_path: str = None,
    eng_column: str = None,
    khm_column: str = None
) -> Dict[str, Any]:
    """
    Converts a CSV file with English-Khmer translation pairs to the format required for NLLB training.
    
    Args:
        csv_file_path: Path to the CSV file containing English-Khmer pairs
        output_file_path: Optional path to save the output as JSON
        eng_column: Column name for English text (if known)
        khm_column: Column name for Khmer text (if known)
        
    Returns:
        A dictionary in the required format for NLLB training
    """
    # Check if the file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"The file {csv_file_path} does not exist.")
    
    # Initialize the result structure
    result = {
        'train': {
            'data': []
        }
    }
    
    # Try to open with utf-8 encoding first
    try:
        encoding = 'utf-8'
        with open(csv_file_path, 'r', encoding=encoding) as f:
            f.read(1024)  # Test read
    except UnicodeDecodeError:
        encoding = 'latin-1'  # Fall back to latin-1
        print(f"Warning: UTF-8 encoding failed, using {encoding} instead.")
    
    # Count lines for progress bar
    with open(csv_file_path, 'r', encoding=encoding) as f:
        total_lines = sum(1 for _ in f)
    
    try:
        with open(csv_file_path, 'r', encoding=encoding) as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get the field names
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError("CSV file has no header row or is empty.")
            
            # Determine which columns to use for English and Khmer if not specified
            if not eng_column:
                # Try exact matches first, then partial matches
                if 'eng_Latn' in fieldnames:
                    eng_column = 'eng_Latn'
                else:
                    for col in fieldnames:
                        if 'eng' in col.lower() or 'english' in col.lower():
                            eng_column = col
                            break
                
                if not eng_column and len(fieldnames) > 0:
                    eng_column = fieldnames[0]  # Default to first column
                    print(f"Warning: Could not identify English column, using first column: '{eng_column}'")
            
            if not khm_column:
                if 'khm_Khmr' in fieldnames:
                    khm_column = 'khm_Khmr'
                else:
                    for col in fieldnames:
                        if 'khm' in col.lower() or 'khmer' in col.lower():
                            khm_column = col
                            break
                
                if not khm_column and len(fieldnames) > 1:
                    khm_column = fieldnames[1]  # Default to second column
                    print(f"Warning: Could not identify Khmer column, using second column: '{khm_column}'")
            
            # Validate that we found the columns
            if not eng_column or not khm_column:
                raise ValueError("Could not identify English and/or Khmer columns in the CSV file.")
            
            if eng_column not in fieldnames:
                raise ValueError(f"Column '{eng_column}' not found in CSV file. Available columns: {', '.join(fieldnames)}")
            
            if khm_column not in fieldnames:
                raise ValueError(f"Column '{khm_column}' not found in CSV file. Available columns: {', '.join(fieldnames)}")
            
            print(f"Using '{eng_column}' for English and '{khm_column}' for Khmer")
            print(f"Processing {total_lines-1} rows...")
            
            valid_rows = 0
            skipped_rows = 0
            
            # Process each row
            for row in tqdm(reader, total=total_lines-1):
                # Check for missing values
                if eng_column not in row or khm_column not in row:
                    skipped_rows += 1
                    continue
                
                eng_text = row[eng_column].strip()
                khm_text = row[khm_column].strip()
                
                # Skip empty entries
                if not eng_text or not khm_text:
                    skipped_rows += 1
                    continue
                
                # Add to the result
                result['train']['data'].append({
                    'eng_Latn': eng_text,
                    'khm_Khmr': khm_text
                })
                valid_rows += 1
            
            print(f"Processed {valid_rows} valid entries. Skipped {skipped_rows} entries with missing data.")
    
    except Exception as e:
        raise Exception(f"Error processing CSV file: {str(e)}")
    
    # Save to file if requested
    if output_file_path:
        try:
            with open(output_file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(result, jsonfile, ensure_ascii=False)
            print(f"Saved result to {output_file_path}")
        except Exception as e:
            raise Exception(f"Error saving to file: {str(e)}")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert a CSV file to the format required for NLLB training.')
    parser.add_argument('csv_file', help='Path to the CSV file containing English-Khmer pairs')
    parser.add_argument('--output', help='Path to save the output JSON file')
    parser.add_argument('--eng-col', help='Column name for English text')
    parser.add_argument('--khm-col', help='Column name for Khmer text')
    
    args = parser.parse_args()
    
    try:
        result = convert_csv_to_nllb_format(
            args.csv_file,
            args.output,
            args.eng_col,
            args.khm_col
        )
        
        # Display sample entries if not saving to file
        if not args.output:
            print("\nSample entries:")
            for i, entry in enumerate(result['train']['data'][:3]):
                print(f"{i+1}. English: {entry['eng_Latn']}")
                print(f"   Khmer: {entry['khm_Khmr']}")
                print()
    except Exception as e:
        print(f"Error: {e}")
