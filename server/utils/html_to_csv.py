#!/usr/bin/env python3
"""
Script to export tracking data from HTML file to CSV format.
Extracts viewsData from HTML JavaScript and converts to tracking.csv format.
"""

import re
import json
import csv
import os
import sys
from typing import Dict, List, Tuple


def extract_views_data_from_html(html_content: str) -> Dict:
    """
    Extract viewsData JavaScript object from HTML content.
    
    Args:
        html_content: HTML file content as string
        
    Returns:
        Dictionary containing viewsData
    """
    # Find the viewsData line - it's a very long line
    pattern = r'const viewsData = ({.+?});'
    match = re.search(pattern, html_content, re.DOTALL)
    
    if not match:
        raise ValueError("Could not find viewsData in HTML file")
    
    # Extract the JavaScript object string
    views_data_str = match.group(1)
    
    # Use ast.literal_eval or manual parsing
    # Since it's a JavaScript object, we need to handle it carefully
    # Try using json.loads first (if it's valid JSON)
    try:
        views_data = json.loads(views_data_str)
    except json.JSONDecodeError:
        # JavaScript objects use double quotes for keys, so it should be JSON-compatible
        # But might have trailing commas or other issues
        # Try to clean it up
        views_data_str_clean = views_data_str.strip()
        
        # Remove trailing commas before closing braces/brackets
        views_data_str_clean = re.sub(r',(\s*[}\]])', r'\1', views_data_str_clean)
        
        try:
            views_data = json.loads(views_data_str_clean)
        except json.JSONDecodeError as e:
            # Last resort: use eval (not recommended but works for trusted data)
            # Replace JavaScript object syntax with Python dict syntax
            # This is a simple approach - be careful with untrusted data
            try:
                # Replace JavaScript null with Python None
                views_data_str_py = views_data_str.replace('null', 'None')
                # Use eval to parse (only for trusted HTML files)
                views_data = eval(views_data_str_py)
            except Exception as eval_error:
                raise ValueError(f"Could not parse viewsData: JSON error: {e}, eval error: {eval_error}")
    
    return views_data


def convert_views_data_to_csv_rows(views_data: Dict) -> List[Tuple[float, str, int, int]]:
    """
    Convert viewsData dictionary to CSV rows format.
    
    Args:
        views_data: Dictionary with view names as keys and data as values
        
    Returns:
        List of tuples: (timestamp, view, slice_number, slice_position)
    """
    rows = []
    
    # Process each view
    for view_name, view_data in views_data.items():
        if not isinstance(view_data, dict):
            continue
            
        times = view_data.get('times', [])
        indices = view_data.get('indices', [])
        
        # Ensure times and indices have same length
        min_length = min(len(times), len(indices))
        times = times[:min_length]
        indices = indices[:min_length]
        
        # Create rows for this view
        for time, index in zip(times, indices):
            rows.append((float(time), view_name, int(index), int(index)))
    
    # Sort by timestamp, then by view name for consistent output
    rows.sort(key=lambda x: (x[0], x[1]))
    
    return rows


def export_html_to_csv(html_file_path: str, output_csv_path: str = None) -> str:
    """
    Export tracking data from HTML file to CSV.
    
    Args:
        html_file_path: Path to input HTML file
        output_csv_path: Path to output CSV file (optional, defaults to same directory)
        
    Returns:
        Path to created CSV file
    """
    # Read HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Extract viewsData
    views_data = extract_views_data_from_html(html_content)
    
    # Convert to CSV rows
    rows = convert_views_data_to_csv_rows(views_data)
    
    # Determine output path
    if output_csv_path is None:
        html_dir = os.path.dirname(html_file_path)
        html_basename = os.path.basename(html_file_path)
        csv_filename = os.path.splitext(html_basename)[0] + '_tracking.csv'
        output_csv_path = os.path.join(html_dir, csv_filename)
    
    # Write CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['timestamp', 'view', 'slice_number', 'slice_position'])
        # Write data rows
        writer.writerows(rows)
    
    print(f"Successfully exported {len(rows)} rows to {output_csv_path}")
    return output_csv_path


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python html_to_csv.py <html_file> [output_csv_file]")
        sys.exit(1)
    
    html_file = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(html_file):
        print(f"Error: HTML file not found: {html_file}")
        sys.exit(1)
    
    try:
        csv_path = export_html_to_csv(html_file, output_csv)
        print(f"Export completed: {csv_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

