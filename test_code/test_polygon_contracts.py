#!/usr/bin/env python
"""Test Polygon futures contracts endpoint to see what's available."""

import os
from polygon import RESTClient

# Get API key
api_key = os.environ.get('POLYGON_API_KEY')
if not api_key:
    print("POLYGON_API_KEY not set")
    exit(1)

client = RESTClient(api_key=api_key)

print("Testing Polygon futures contracts endpoint:")
print("=" * 80)

try:
    # Try to list futures contracts for ES
    print("\nAttempting to list ES futures contracts...")

    # The polygon-api-client may have a list_snapshot_all method
    # or we might need to use the underlying REST API

    # Try different approaches
    methods_to_try = [
        'list_futures_contracts',
        'get_futures_contracts',
        'list_snapshot_all',
    ]

    for method_name in methods_to_try:
        if hasattr(client, method_name):
            print(f"\n✓ Found method: {method_name}")
            method = getattr(client, method_name)
            print(f"  Signature: {method.__doc__}")
        else:
            print(f"\n✗ Method not found: {method_name}")

    # List all available methods that might be relevant
    print("\n\nAll client methods containing 'futures' or 'contract':")
    for attr in dir(client):
        if 'future' in attr.lower() or 'contract' in attr.lower():
            print(f"  - {attr}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
