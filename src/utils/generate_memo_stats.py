import pandas as pd
from pathlib import Path

def generate_strategy_memo_offenders(min_trips: int = 50) -> pd.DataFrame:
    """
    Identifies the top 5 chronically delayed corridors responsible for the highest 
    volume of SLA breaches, formatting the output for the Executive Strategy Memo.
    """
    # 1. Robust Path Handling (No more "../")
    data_path = Path("data/processed/trips_clean.parquet")
    
    # Fallback just in case you saved it in data/ instead of data/processed/
    if not data_path.exists():
        data_path = Path("data/trips_clean.parquet")
        
    if not data_path.exists():
        print(f"❌ ERROR: Could not find trips_clean.parquet in data/ or data/processed/.")
        print("Please ensure your Phase 1 Jupyter Notebook has successfully created this file.")
        return pd.DataFrame()
    
    # 2. Load the cleaned Phase 1 data
    df = pd.read_parquet(data_path)
    
    # 3. Define SLA Breach 
    df['is_sla_breach'] = df['factor'] > 1.2
    
    # 4. Aggregate at the corridor level
    offenders = df.groupby(['source_name', 'destination_name']).agg(
        total_trips=('trip_uuid', 'count'),
        breach_count=('is_sla_breach', 'sum'),
        median_delay_ratio=('factor', 'median')
    ).reset_index()
    
    # 5. Filter out statistically insignificant corridors
    offenders = offenders[offenders['total_trips'] >= min_trips]
    
    # 6. Calculate percentage contribution to overall network failure
    total_network_breaches = df['is_sla_breach'].sum()
    offenders['pct_of_network_breaches'] = (offenders['breach_count'] / total_network_breaches) * 100
    
    # 7. Isolate the top 5
    top_5 = offenders.sort_values(by='breach_count', ascending=False).head(5)
    
    # Print Memo-Ready Output to Terminal
    print("=" * 80)
    print("🏆 STRATEGY MEMO: TOP 5 OFFENDER CORRIDORS")
    print("=" * 80)
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {row['source_name']} ➔ {row['destination_name']}")
        print(f"   • SLA Breaches: {row['breach_count']:,} (out of {row['total_trips']:,} total trips)")
        print(f"   • Contribution to Network Failure: {row['pct_of_network_breaches']:.2f}%")
        print(f"   • Median Delay Ratio: {row['median_delay_ratio']:.2f}x slower than OSRM")
        print("-" * 80)
        
    # 8. File-Saving Logic (Fixed Path)
    output_path = Path("reports/top_5_offenders.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_5.to_csv(output_path, index=False)
    
    print(f"✅ SUCCESS: Top 5 offenders saved to {output_path}")
    return top_5

# Execute the script when run directly
if __name__ == "__main__":
    generate_strategy_memo_offenders()