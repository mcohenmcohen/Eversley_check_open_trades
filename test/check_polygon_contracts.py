#!/usr/bin/env python3

import requests
import json

def get_polygon_futures_contracts():
    """Get all futures contracts from Polygon API"""

    api_key = "qFTWmhmyAj2pJqpx0Pwxp2AyShKFVPv9"
    url = f"https://api.polygon.io/futures/vX/contracts?active=all&type=all&limit=1000&sort=product_code.asc&apikey={api_key}"

    print("🔍 Fetching all futures contracts from Polygon...")

    try:
        response = requests.get(url)
        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if 'results' in data:
                contracts = data['results']
                print(f"Found {len(contracts)} contracts in this batch")

                # Check if we hit the limit
                total_count = data.get('count', len(contracts))
                next_url = data.get('next_url', None)

                print(f"Total available: {total_count}")
                print(f"Next URL available: {'Yes' if next_url else 'No'}")

                if len(contracts) >= 1000 and next_url:
                    print("⚠️  Hit 1000 limit - there are more contracts available!")
                    print("Let me fetch more batches...")

                    # Fetch additional batches
                    all_contracts = contracts.copy()
                    current_url = next_url
                    batch_count = 1

                    while current_url and batch_count < 10:  # Limit to 10 batches (10k contracts)
                        print(f"Fetching batch {batch_count + 1}...")
                        try:
                            batch_response = requests.get(current_url)
                            if batch_response.status_code == 200:
                                batch_data = batch_response.json()
                                if 'results' in batch_data:
                                    batch_contracts = batch_data['results']
                                    all_contracts.extend(batch_contracts)
                                    current_url = batch_data.get('next_url', None)
                                    print(f"  Added {len(batch_contracts)} more contracts")
                                    batch_count += 1
                                else:
                                    break
                            else:
                                print(f"  Error fetching batch: {batch_response.status_code}")
                                break
                        except Exception as e:
                            print(f"  Error fetching batch: {e}")
                            break

                    contracts = all_contracts
                    print(f"Total contracts fetched: {len(contracts)}")

                contracts = contracts

                # Look for our target symbols
                target_symbols = ['6A', '6B', '6C', '6E', '6S', 'ES', 'NQ', 'YM', 'RTY']
                found_contracts = {}

                for target in target_symbols:
                    found_contracts[target] = []

                print(f"\nSearching for contracts containing: {target_symbols}")
                print("=" * 60)

                for contract in contracts:
                    ticker = contract.get('ticker', '')
                    product_code = contract.get('product_code', '')
                    underlying_ticker = contract.get('underlying_ticker', '')
                    name = contract.get('name', '')
                    expiration = contract.get('expiration_date', '')

                    # Check if this contract matches any of our targets
                    for target in target_symbols:
                        # More flexible matching
                        matches = (
                            target in ticker or
                            target in product_code or
                            target in underlying_ticker or
                            target in name.upper() or
                            # Also check for variations
                            ticker.startswith(target) or
                            ticker.endswith(target) or
                            # Check for currency names
                            (target == '6A' and any(x in name.upper() for x in ['AUSTRALIAN', 'AUD'])) or
                            (target == '6B' and any(x in name.upper() for x in ['BRITISH', 'GBP', 'POUND'])) or
                            (target == '6C' and any(x in name.upper() for x in ['CANADIAN', 'CAD'])) or
                            (target == '6E' and any(x in name.upper() for x in ['EURO', 'EUR'])) or
                            (target == '6S' and any(x in name.upper() for x in ['SWISS', 'CHF', 'FRANC'])) or
                            # Check for index names
                            (target == 'ES' and any(x in name.upper() for x in ['S&P', 'SP500', 'S&P 500'])) or
                            (target == 'NQ' and any(x in name.upper() for x in ['NASDAQ', 'NDX', 'NASDAQ-100'])) or
                            (target == 'YM' and any(x in name.upper() for x in ['DOW', 'DJIA', 'DOW JONES'])) or
                            (target == 'RTY' and any(x in name.upper() for x in ['RUSSELL', 'RTY', 'RUSSELL 2000']))
                        )

                        if matches:
                            found_contracts[target].append({
                                'ticker': ticker,
                                'product_code': product_code,
                                'underlying_ticker': underlying_ticker,
                                'name': name,
                                'expiration': expiration
                            })

                # Display results
                for target, contracts in found_contracts.items():
                    print(f"\n📊 {target} Contracts:")
                    if contracts:
                        for contract in contracts[:5]:  # Show first 5
                            print(f"  ✅ {contract['ticker']} - {contract['name']}")
                            print(f"     Product: {contract['product_code']}, Expires: {contract['expiration']}")
                        if len(contracts) > 5:
                            print(f"     ... and {len(contracts)-5} more")
                    else:
                        print(f"  ❌ No {target} contracts found")

                # Also show a sample of what IS available
                print(f"\n📋 Sample of available contracts (first 10):")
                for i, contract in enumerate(contracts[:10]):
                    print(f"  {i+1}. {contract.get('ticker', 'N/A')} - {contract.get('name', 'N/A')}")
                    print(f"     Product: {contract.get('product_code', 'N/A')}")

                return found_contracts

            else:
                print("❌ No 'results' field in response")
                print(f"Response keys: {data.keys()}")
                return None

        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_found_contracts(found_contracts):
    """Test if any found contracts actually work for data retrieval"""

    if not found_contracts:
        print("No contracts to test")
        return

    print(f"\n🧪 Testing data retrieval for found contracts...")

    # We'll need to use the polygon-api-client here, but let's show what we'd test
    working_tickers = []

    for target, contracts in found_contracts.items():
        if contracts:
            # Test the first contract for each symbol
            test_ticker = contracts[0]['ticker']
            print(f"  Would test: {test_ticker} for {target}")
            working_tickers.append(test_ticker)

    return working_tickers

if __name__ == "__main__":
    print("🚀 Polygon Futures Contract Checker")
    print("=" * 50)

    found = get_polygon_futures_contracts()

    if found:
        working = test_found_contracts(found)

        if any(contracts for contracts in found.values()):
            print(f"\n🎉 Found some matching contracts!")
            print("You can use these tickers in your bulk download script:")
            for target, contracts in found.items():
                if contracts:
                    print(f"  {target}: {contracts[0]['ticker']}")
        else:
            print(f"\n❌ No matching contracts found for currency/index futures")
            print("This suggests:")
            print("1. Polygon may not have these specific futures")
            print("2. Different symbol naming convention")
            print("3. Need different API access level")
    else:
        print("❌ Failed to get contract data")