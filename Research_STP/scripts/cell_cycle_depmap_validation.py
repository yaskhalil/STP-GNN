import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def validate_cell_cycle():
    print("="*80)
    print("DEPMAP EMPIRICAL VALIDATION: 10-NODE CELL CYCLE MODEL")
    print("="*80)
    
    file_path = "depmap_data/CRISPRGeneEffect.csv"
    
    # Map model nodes to DepMap columns
    mapping = {
        "CycD": "CCND1 (595)",
        "Rb": "RB1 (5925)",
        "E2F": "E2F1 (1869)",
        "CycE": "CCNE1 (898)",
        "CycA": "CCNA2 (890)",
        "p27": "CDKN1B (1027)",
        "Cdc20": "CDC20 (991)",
        "Cdh1": "FZR1 (51343)",
        "UbcH10": "UBE2C (11065)",
        "CycB": "CCNB1 (891)"
    }
    
    # Load data
    print("Loading DepMap CRISPR dataset...")
    # The first column is unnamed (ModelID)
    cols = list(mapping.values())
    df = pd.read_csv(file_path, usecols=cols)
    df.rename(columns={v: k for k, v in mapping.items()}, inplace=True)
    df.dropna(inplace=True)
    
    # 1. Dependency Profile Analysis
    # In DepMap, scores < -0.5 or -1.0 indicate "essentiality" (the cell dies if knocked out).
    essentiality = (df.iloc[:, 1:] < -0.5).mean()
    print("\n[ESSENTIALITY RANKING] (Percentage of cell lines dependent):")
    for gene, score in essentiality.sort_values(ascending=False).items():
        print(f"  {gene:<10}: {score:.2%}")

    # 2. Bottleneck Validation (E2F vs CycD)
    # Our model said E2F and CycD were the "hijacked" transitions.
    r, p = stats.pearsonr(df['E2F'], df['CycD'])
    print(f"\n[BOTTLENECK VALIDATION] E2F vs CycD Correlation:")
    print(f"  Pearson r: {r:.4f} (p={p:.4e})")

    # 3. Regulatory Logic Verification: Rb-E2F Axis
    # Theory: High Rb (inhibitor) should correlate with LOW dependency on E2F (if Rb is working).
    # If Rb is lost, the cell becomes hyper-dependent on E2F.
    rb_wt = df[df['Rb'] > -0.2] # Rb functional
    rb_loss = df[df['Rb'] < -0.5] # Rb loss
    
    print(f"\n[REGULATORY LOGIC] Rb-E2F Axis Stratification:")
    print(f"  Mean E2F Dependency (Rb-WT): {rb_wt['E2F'].mean():.4f}")
    print(f"  Mean E2F Dependency (Rb-Loss): {rb_loss['E2F'].mean():.4f}")

    # Visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.iloc[:, 1:].corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title("Empirical Correlation Matrix: 10-Node Cell Cycle Nodes")
    plt.savefig('cell_cycle_depmap_corr.png')
    print("\nCorrelation matrix saved to: cell_cycle_depmap_corr.png")

    # Conclusion
    if rb_loss['E2F'].mean() < rb_wt['E2F'].mean():
        print("\nVERDICT: EMPIRICAL PARITY CONFIRMED. The hyper-dependency on E2F in Rb-loss contexts validates the STP-GNN's identification of the E2F bottleneck.")
    else:
        print("\nVERDICT: DISCREPANCY DETECTED. Real-world feedback loops may be more complex than the 10-node model.")

if __name__ == "__main__":
    validate_cell_cycle()
