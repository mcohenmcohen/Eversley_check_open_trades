#!/usr/bin/env python3
"""
Find correct InsightSentry futures symbol mappings using their search API.
"""

import os
import requests
import json

# API Key from environment
INSIGHTSENTRY_API_KEY = os.getenv('INSIGHTSENTRY_API_KEY')

if not INSIGHTSENTRY_API_KEY:
    print("❌ INSIGHTSENTRY_API_KEY environment variable not set")
    exit(1)

BASE_URL = "https://api.insightsentry.com"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {INSIGHTSENTRY_API_KEY}'
}

def search_symbols(query, symbol_type="futures", page=1, limit=10):
    """Search for symbols using InsightSentry v3 API."""
    try:
        url = f"{BASE_URL}/v3/symbols/search"
        params = {
            'query': query,
            'type': symbol_type,  # Try 'futures', 'stock', 'index', etc.
            'page': page
        }
        
        print(f"🔍 Searching '{query}' (type: {symbol_type})")
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None

def main():
    """Search for all our futures symbols."""
    print("🚀 InsightSentry Futures Symbol Search")
    print("=" * 60)
    
    # Our target symbols
    search_terms = {
        # Currency futures
        '6A': ['australian dollar', 'aud', '6A', 'AUD/USD'],
        '6B': ['british pound', 'gbp', '6B', 'GBP/USD'],
        '6C': ['canadian dollar', 'cad', '6C', 'USD/CAD'],
        '6E': ['euro', 'eur', '6E', 'EUR/USD'],
        '6S': ['swiss franc', 'chf', '6S', 'USD/CHF'],
        
        # Index futures
        'ES': ['e-mini s&p', 'sp500', 'ES', 's&p 500'],
        'NQ': ['e-mini nasdaq', 'nasdaq', 'NQ', 'nasdaq 100'],
        'RTY': ['russell 2000', 'RTY', 'russell'],
        'YM': ['dow jones', 'YM', 'djia', 'dow']
    }
    
    found_mappings = {}
    
    # Try different types
    symbol_types = ['futures', 'stock', 'index', 'forex', 'commodity']
    
    for internal_symbol, queries in search_terms.items():
        print(f"\n{'='*20} {internal_symbol} {'='*20}")
        
        best_matches = []
        
        for symbol_type in symbol_types:
            for query in queries:
                print(f"\n🔍 Searching: '{query}' (type: {symbol_type})")
                
                results = search_symbols(query, symbol_type)
                if not results:
                    continue
                
                # Handle response format
                if 'data' in results:
                    items = results['data']
                elif 'results' in results:
                    items = results['results']
                elif isinstance(results, list):
                    items = results
                else:
                    print(f"Unknown response format: {list(results.keys())}")
                    continue
                
                if not items:
                    print("  No results found")
                    continue
                
                print(f"  ✅ Found {len(items)} results:")
                
                for i, item in enumerate(items[:5]):  # Show top 5
                    if isinstance(item, dict):
                        symbol = item.get('symbol', item.get('ticker', item.get('id', 'Unknown')))
                        name = item.get('name', item.get('description', item.get('company_name', 'No name')))
                        exchange = item.get('exchange', item.get('market', ''))
                        
                        print(f"    {i+1}. {symbol:15} {name[:40]:<40} [{exchange}]")
                        
                        # Score potential matches
                        score = 0
                        symbol_lower = symbol.lower()
                        name_lower = name.lower()
                        
                        # Exact symbol match gets highest score
                        if internal_symbol.lower() == symbol_lower:
                            score += 100
                        elif internal_symbol.lower() in symbol_lower:
                            score += 50
                        
                        # Name relevance scoring
                        for search_term in queries:
                            if search_term.lower() in name_lower:
                                score += 10
                        
                        # Futures/derivatives keywords
                        futures_keywords = ['future', 'mini', 'e-mini', 'continuous', 'front month']
                        for keyword in futures_keywords:
                            if keyword in name_lower:
                                score += 5
                        
                        best_matches.append((score, symbol, name, exchange, symbol_type))
        
        # Show best matches for this symbol
        if best_matches:
            best_matches.sort(key=lambda x: x[0], reverse=True)
            print(f"\n🏆 Best matches for {internal_symbol}:")
            for score, symbol, name, exchange, stype in best_matches[:3]:
                print(f"    {score:3d} pts: {symbol:15} {name[:40]:<40} [{exchange}] ({stype})")
            
            # Auto-select best match
            top_match = best_matches[0]
            if top_match[0] > 10:  # Minimum score threshold
                found_mappings[internal_symbol] = top_match[1]
        
        print("-" * 60)
    
    # Final summary
    print(f"\n📊 FINAL SYMBOL MAPPINGS:")
    print("=" * 60)
    
    if found_mappings:
        print("# Update these mappings in data_sources.py:")
        print("futures_mappings = {")
        for internal, external in found_mappings.items():
            print(f"    '{internal}': '{external}',  # {search_terms[internal][0].title()}")
        print("}")
        
        print(f"\n✅ Found {len(found_mappings)} out of {len(search_terms)} symbols")
    else:
        print("❌ No reliable symbol mappings found")
    
    print(f"\n💡 Manual verification recommended for each symbol")

if __name__ == "__main__":
    main()