"""
Merge test_tables.json into tables.json, then rebuild FAISS index.
This ensures all test database schemas are included in the vector store.

Usage:
    python merge_test_tables.py
"""

import json
import subprocess
import os

def main():
    print("\n" + "=" * 60)
    print("MERGE TEST TABLES AND REBUILD FAISS INDEX")
    print("=" * 60)
    
    # Step 1: Merge tables.json with test_tables.json
    print("\nStep 1: Merging tables.json with test_tables.json...")
    
    with open('spider_data/tables.json', 'r') as f:
        tables = json.load(f)
    
    with open('spider_data/test_tables.json', 'r') as f:
        test_tables = json.load(f)
    
    existing_dbs = {t['db_id'] for t in tables}
    new_tables = [t for t in test_tables if t['db_id'] not in existing_dbs]
    
    print(f'  Train/dev tables: {len(tables)}')
    print(f'  Test tables: {len(test_tables)}')
    print(f'  New unique test tables: {len(new_tables)}')
    
    merged = tables + new_tables
    print(f'  Total merged: {len(merged)}')
    
    with open('spider_data/tables.json', 'w') as f:
        json.dump(merged, f, indent=2)
    
    print('✓ Merged and saved to spider_data/tables.json')
    
    # Step 2: Backup existing FAISS files
    print("\nStep 2: Backing up existing FAISS files...")
    if os.path.exists('output/spider_processed.json'):
        subprocess.run(['cp', 'output/spider_processed.json', 'output/spider_processed_backup.json'])
        print('✓ Backed up spider_processed.json')
    
    if os.path.exists('output/faiss.index'):
        subprocess.run(['cp', 'output/faiss.index', 'output/faiss_backup.index'])
        print('✓ Backed up faiss.index')
    
    # Step 3: Reprocess Spider data
    print("\nStep 3: Running preprocess_spider.py...")
    result = subprocess.run([
        './venv/bin/python', 'preprocess_spider.py',
        '--spider-dir', 'spider_data',
        '--database-dir', 'spider_data/database'
    ])
    
    if result.returncode != 0:
        print("❌ Error running preprocess_spider.py")
        return
    
    # Step 4: Rebuild FAISS index
    print("\nStep 4: Running ingest.py...")
    result = subprocess.run([
        './venv/bin/python', 'ingest.py',
        '--input', 'output/spider_processed.json'
    ])
    
    if result.returncode != 0:
        print("❌ Error running ingest.py")
        return
    
    # Step 5: Verify the results
    print("\nStep 5: Verifying results...")
    with open('output/metadata.json', 'r') as f:
        meta = json.load(f)
    
    schemas = [m for m in meta if m.get('type') == 'schema']
    print(f'✓ Total schemas in index: {len(schemas)}')
    print(f'✓ Total entries in index: {len(meta)}')
    
    print("\n" + "=" * 60)
    print("✓ MERGE AND REBUILD COMPLETE!")
    print("=" * 60)
    print(f"Added {len(new_tables)} new test database schemas")
    print(f"Total schemas: {len(tables)} → {len(schemas)}")
    print("Backup files: spider_processed_backup.json, faiss_backup.index")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
