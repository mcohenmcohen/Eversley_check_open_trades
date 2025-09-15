#!/usr/bin/env python3
"""
Test the 9 futures symbols using InsightSentry's search API.
Format example: CME_MINI:NQU2025
"""

import os
import requests
import time

INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')
BASE_URL = "https://api.insightsentry.com"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {INSIGHTSENTRY_API_KEY}'
}

def search_symbol(query, symbol_type="futures"):
    """Search using the v3 API endpoint."""
    try:
        url = f"{BASE_URL}/v3/symbols/search"
        params = {
            'query': query,
            'type': symbol_type
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"⚠️ Rate limit hit for {query}, waiting...")
            time.sleep(2)
            return None
        else:
            print(f"❌ Error {response.status_code} for {query}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error searching {query}: {e}")
        return None

def main():
    # Our 9 symbols
    test_symbols = ['6A', '6B', '6C', '6E', '6S', 'ES', 'NQ', 'RTY', 'YM']
    
    print("🔍 Testing 9 Futures Symbols in InsightSentry")
    print("=" * 60)
    
    found_mappings = {}
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        # Search for the symbol
        results = search_symbol(symbol, "futures")
        if not results:
            time.sleep(1)  # Rate limit protection
            continue
        
        # Extract symbols from response
        symbols = results.get('symbols', [])
        if not symbols:
            print(f"  No symbols found for {symbol}")
            continue
        
        print(f"  Found {len(symbols)} results:")
        
        # Show top results and look for best match
        best_match = None
        best_score = 0
        
        for i, item in enumerate(symbols[:5]):  # Top 5 results
            sym = item.get('symbol', 'Unknown')
            name = item.get('name', 'No name')
            exchange = item.get('exchange', '')
            
            print(f"    {i+1}. {sym:25} {name[:35]:<35} [{exchange}]")
            
            # Score this match
            score = 0
            if symbol.upper() in sym.upper():
                score += 10
            if 'CME' in sym or 'MINI' in sym:
                score += 5
            if any(word in name.lower() for word in ['future', 'continuous', 'mini']):
                score += 3
            
            if score > best_score:
                best_score = score
                best_match = sym
        
        if best_match:
            found_mappings[symbol] = best_match
            print(f"  🏆 Best match: {best_match} (score: {best_score})")
        
        time.sleep(0.5)  # Rate limit protection
    
    print(f"\n📊 FINAL MAPPINGS:")
    print("=" * 60)
    
    if found_mappings:
        print("# Copy this to data_sources.py futures_mappings:")
        print("futures_mappings = {")
        for internal, external in found_mappings.items():
            print(f"    '{internal}': '{external}',")
        print("}")
        
        print(f"\n✅ Found {len(found_mappings)}/9 symbols")
    else:
        print("❌ No symbols found")

if __name__ == "__main__":
    if not INSIGHTSENTRY_API_KEY:
        print("❌ INSIGHTSENTRY_API_KEY not set")
        exit(1)
    main()