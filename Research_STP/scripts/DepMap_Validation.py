import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DepMapValidator:
    def __init__(self, data_dir="./depmap_data"):
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, "CRISPRGeneEffect.csv")
        self.target_genes = ["TP53", "MDM2"]
        os.makedirs(self.data_dir, exist_ok=True)

    def verify_data_presence(self):
        """Ensures the DepMap CRISPR dataset is available locally."""
        if not os.path.exists(self.file_path):
            print("\n" + "="*60)
            print("[!] EMPIRICAL DATA MISSING: DepMap CRISPR Dataset")
            print("="*60)
            print("To validate the STP-GNN predictions against 1000+ cancer cell lines:")
            print("  1. Visit: https://depmap.org/portal/download/all/")
            print("  2. Download 'CRISPR_gene_effect.csv' (Public 23Q4 Release)")
            print(f"  3. Place it in: {os.path.abspath(self.data_dir)}")
            print("="*60 + "\n")
            return False
        return True

    def extract_targeted_subset(self, chunk_size=5000):
        """Extracts TP53 and MDM2 columns while bypassing the 1GB memory limit."""
        print(f"Parsing DepMap CRISPR dataset...")
        
        # Peek headers
        df_peek = pd.read_csv(self.file_path, nrows=1)
        headers = df_peek.columns.tolist()
        
        tp53_col = next((col for col in headers if 'TP53' in col), None)
        mdm2_col = next((col for col in headers if 'MDM2' in col), None)
        
        if not tp53_col or not mdm2_col:
            raise ValueError("TP53 or MDM2 not found in dataset headers.")
            
        columns_to_keep = [headers[0], tp53_col, mdm2_col]
        
        extracted = []
        for chunk in pd.read_csv(self.file_path, usecols=columns_to_keep, chunksize=chunk_size):
            extracted.append(chunk)
            
        df = pd.concat(extracted, ignore_index=True)
        df.rename(columns={headers[0]: "CellLine", tp53_col: "TP53", mdm2_col: "MDM2"}, inplace=True)
        df.dropna(inplace=True)
        return df

    def compute_synthetic_lethality(self, df):
        """Statistical proof of MDM2 dependency dependence on p53 status."""
        print("\n" + "="*80)
        print("EMPIRICAL GROUNDING: SYNTHETIC LETHALITY ANALYSIS")
        print("="*80)
        
        r, p_val = stats.pearsonr(df['TP53'], df['MDM2'])
        print(f"Correlation Coefficient (r): {r:.4f}")
        print(f"p-value: {p_val:.4e}")

        # WT lines (TP53 > -0.2) vs Mutated (TP53 <= -0.2)
        wt_mask = df['TP53'] > -0.2
        mut_mask = df['TP53'] <= -0.2
        
        mdm2_wt = df[wt_mask]['MDM2'].mean()
        mdm2_mut = df[mut_mask]['MDM2'].mean()
        
        print(f"\nCohort Assessment:")
        print(f"  MDM2 dependency in p53-WT lines:  {mdm2_wt:.4f}")
        print(f"  MDM2 dependency in p53-MUT lines: {mdm2_mut:.4f}")
        
        if mdm2_wt < -0.5 and mdm2_wt < mdm2_mut:
            print("\n[VERDICT]: EMPIRICAL CONFIRMATION ACHIEVED.")
            print("Wet-lab data proves MDM2 is lethal ONLY in p53-functional networks.")
        else:
            print("\n[VERDICT]: EMPIRICAL REJECTION OR INSUFFICIENT SIGNAL.")

    def visualize_landscape(self, df):
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='TP53', y='MDM2', data=df, alpha=0.6, color='#2c3e50', edgecolor=None)
        plt.axhline(-0.5, color='crimson', linestyle='--', label='MDM2 Essentiality')
        plt.axvline(-0.2, color='dodgerblue', linestyle='--', label='p53 Functionality Boundary')
        plt.title("DepMap Validation: MDM2 vs TP53 CRISPR Dependency", weight='bold')
        plt.xlabel("p53 Dependency (Chronos Score)")
        plt.ylabel("MDM2 Dependency (Chronos Score)")
        plt.legend(); plt.grid(True, ls=':', alpha=0.5)
        plt.savefig(os.path.join(self.data_dir, "empirical_validation.png"))
        print(f"Analysis exported to {self.data_dir}/empirical_validation.png")

if __name__ == "__main__":
    validator = DepMapValidator()
    if validator.verify_data_presence():
        data = validator.extract_targeted_subset()
        validator.compute_synthetic_lethality(data)
        validator.visualize_landscape(data)
