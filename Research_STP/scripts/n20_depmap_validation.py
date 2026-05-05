import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

class N20DepMapValidator:
    def __init__(self, data_path="depmap_data/CRISPRGeneEffect.csv"):
        self.data_path = data_path
        self.target_cols = {
            "TP53": "TP53 (7157)",
            "DICER1": "DICER1 (23405)",
            "ZEB1": "ZEB1 (6935)"
        }

    def load_data(self):
        # Read only necessary columns
        headers = pd.read_csv(self.data_path, nrows=0).columns.tolist()
        cols_to_use = [headers[0]] + list(self.target_cols.values())
        df = pd.read_csv(self.data_path, usecols=cols_to_use)
        df.rename(columns={
            self.target_cols["TP53"]: "TP53",
            self.target_cols["DICER1"]: "DICER1",
            self.target_cols["ZEB1"]: "ZEB1"
        }, inplace=True)
        return df.dropna()

    def analyze(self, df):
        print("="*80)
        print("N=20 EMPIRICAL VALIDATION: MULTIVARIATE CRISPR ANALYSIS")
        print("="*80)
        
        # Cohort Stratification
        wt_df = df[df['TP53'] > -0.2]
        mut_df = df[df['TP53'] <= -0.2]
        
        # Correlation in WT cohort
        r_dz_wt, p_dz_wt = stats.pearsonr(wt_df['DICER1'], wt_df['ZEB1'])
        
        print(f"TP53-WildType Cohort (N={len(wt_df)}):")
        print(f"  DICER1-ZEB1 Correlation (r): {r_dz_wt:.4f}")
        print(f"  p-value: {p_dz_wt:.4e}")
        
        print(f"\nTP53-Mutated Cohort (N={len(mut_df)}):")
        r_dz_mut, _ = stats.pearsonr(mut_df['DICER1'], mut_df['ZEB1'])
        print(f"  DICER1-ZEB1 Correlation (r): {r_dz_mut:.4f}")
        
        return r_dz_wt

    def visualize_3d(self, df):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color based on TP53 functionality
        colors = ['dodgerblue' if x > -0.2 else 'crimson' for x in df['TP53']]
        
        scatter = ax.scatter(df['TP53'], df['DICER1'], df['ZEB1'], 
                            c=colors, alpha=0.6, s=40, edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel('TP53 (Chronos Score)')
        ax.set_ylabel('DICER1 (Chronos Score)')
        ax.set_zlabel('ZEB1 (Chronos Score)')
        ax.set_title('3D CRISPR Dependency Landscape: TP53 vs DICER1 vs ZEB1', weight='bold')
        
        # Add legend proxies
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='TP53-WT', markerfacecolor='dodgerblue', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='TP53-MUT', markerfacecolor='crimson', markersize=10)]
        ax.legend(handles=legend_elements)
        
        plt.savefig('n20_empirical_landscape.png', dpi=300)
        print("\n3D Landscape Visualization saved to: n20_empirical_landscape.png")

if __name__ == "__main__":
    validator = N20DepMapValidator()
    data = validator.load_data()
    validator.analyze(data)
    validator.visualize_3d(data)
