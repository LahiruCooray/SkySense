#!/usr/bin/env python3
"""
Cleanup utility for SkySense insights
Removes duplicate analyses, keeping only the latest for each log
"""

import os
import glob
import re
from pathlib import Path
from collections import defaultdict


def cleanup_old_analyses(insights_dir="data/insights", dry_run=False):
    """
    Clean up old duplicate analyses, keep only the latest for each log.
    
    Args:
        insights_dir: Path to insights directory
        dry_run: If True, only print what would be deleted
    """
    
    insights_path = Path(insights_dir)
    
    # Find all index files
    index_files = list(insights_path.glob("index_*_*.json"))
    
    if not index_files:
        print("No index files found.")
        return
    
    # Group by log name
    log_groups = defaultdict(list)
    
    for index_file in index_files:
        # Parse filename: index_<log_name>_<timestamp>.json
        match = re.match(r'index_(.+?)_(\d{8}_\d{6})\.(json|parquet)', index_file.name)
        if match:
            log_name = match.group(1)
            timestamp = match.group(2)
            log_groups[log_name].append((timestamp, index_file))
    
    # Process each log group
    total_removed = 0
    total_kept = 0
    
    for log_name, files in log_groups.items():
        if len(files) <= 1:
            total_kept += len(files)
            continue
        
        # Sort by timestamp (newest first)
        files.sort(key=lambda x: x[0], reverse=True)
        
        # Keep the newest
        newest_timestamp, newest_file = files[0]
        print(f"\n📊 Log: {log_name}")
        print(f"   ✅ Keeping: {newest_file.name}")
        total_kept += 1
        
        # Remove older ones
        for old_timestamp, old_file in files[1:]:
            # Also remove corresponding parquet file
            json_file = old_file.with_suffix('.json')
            parquet_file = old_file.with_suffix('.parquet')
            
            for file_to_remove in [json_file, parquet_file]:
                if file_to_remove.exists():
                    if dry_run:
                        print(f"   🗑️  Would remove: {file_to_remove.name}")
                    else:
                        file_to_remove.unlink()
                        print(f"   🗑️  Removed: {file_to_remove.name}")
                    total_removed += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Kept:    {total_kept} analyses")
    print(f"  Removed: {total_removed} files")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up duplicate SkySense analyses")
    parser.add_argument("--insights-dir", default="data/insights", 
                       help="Path to insights directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    print("🧹 SkySense Insights Cleanup")
    print("="*60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be deleted\n")
    
    cleanup_old_analyses(args.insights_dir, args.dry_run)
