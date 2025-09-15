#!/usr/bin/env python3
"""
Find correct InsightSentry symbol formats for ETFs using their search API.
"""
import os
import requests
import json

# API setup
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')
base_url = "https://api.insightsentry.com"
headers = {
    'Authorization': f'Bearer {INSIGHTSENTRY_API_KEY}',
    'Content-Type': 'application/json'
}

def search_symbol(symbol):
    """Search for a symbol using InsightSentry's search API."""
    try:
        url = f"{base_url}/v3/symbols/search"
        params = {
            'query': symbol,
            'type': 'etf',  # Only search for ETFs
            'country': 'US',  # US ETFs only
            'limit': 10
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"❌ Search failed for {symbol}: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"💥 Error searching {symbol}: {e}")
        return []

def main():
    # ETFs that are failing
    failing_etfs = [
        'SPY', 'XLE', 'XLV', 'GDX', 'EFA', 'USO', 'XLB', 'XLF', 'XLK', 
        'XLI', 'XRT', 'XOP', 'XME', 'VNQ', 'IYR', 'XHB', 'IBB', 'TLT',
        'EEM', 'VWO', 'FXI', 'EWG', 'EWH', 'UNG', 'DIA', 'IWM', 'GLD', 'SLV'
    ]
    
    print("🔍 Searching for correct InsightSentry ETF symbol formats...")
    
    symbol_mappings = {}
    
    for etf in failing_etfs:
        print(f"\n🔎 Searching for {etf}...")
        results = search_symbol(etf)
        
        if results:
            # Look for exact matches first
            exact_matches = [r for r in results if r.get('symbol', '').endswith(f':{etf}')]
            if exact_matches:
                best_match = exact_matches[0]
                symbol_mappings[etf] = best_match['symbol']
                print(f"✅ {etf} -> {best_match['symbol']} ({best_match.get('name', 'Unknown')})")
            else:
                # Show all results for manual review
                print(f"⚠️ Multiple results for {etf}:")
                for i, result in enumerate(results[:3]):  # Show top 3
                    print(f"   {i+1}. {result['symbol']} - {result.get('name', 'Unknown')}")
        else:
            print(f"❌ No results found for {etf}")
    
    print(f"\n📋 Final symbol mappings:")
    print("symbol_mappings = {")
    for symbol, mapped in symbol_mappings.items():
        print(f"    '{symbol}': '{mapped}',")
    print("}")
    
    print(f"\n✅ Found mappings for {len(symbol_mappings)}/{len(failing_etfs)} ETFs")

if __name__ == "__main__":
    main()