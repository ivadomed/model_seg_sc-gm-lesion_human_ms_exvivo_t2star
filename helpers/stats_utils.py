import pandas as pd
import numpy as np
from collections import defaultdict

class MetricTracker:
    def __init__(self):
        # Stores TP/FP/FN for global Dice calculation (Dataset-level Dice)
        self.global_counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
        
        # Stores row-by-row metrics (Case-level or Slice-level)
        self.case_metrics = []
        
    def update_counts(self, pred, gt, labels=[1, 2, 3, 4]):
        """
        Accumulates TP/FP/FN counts for the entire dataset.
        Args:
            pred: Prediction mask (integer labels)
            gt: Ground truth mask (integer labels)
            labels: List of class labels to track
        """
        for label in labels:
            p_bin = (pred == label)
            g_bin = (gt == label)
            
            self.global_counts[label]['TP'] += np.logical_and(p_bin, g_bin).sum()
            self.global_counts[label]['FP'] += np.logical_and(p_bin, ~g_bin).sum()
            self.global_counts[label]['FN'] += np.logical_and(~p_bin, g_bin).sum()

    def update_counts_combo(self, pred, gt, labels_to_combine, combo_name):
        """
        Accumulates counts for a combined class (e.g., labels [3,4] treated as one).
        """ 
        p_mask = np.isin(pred, labels_to_combine)
        g_mask = np.isin(gt, labels_to_combine)
        
        self.global_counts[combo_name]['TP'] += np.logical_and(p_mask, g_mask).sum()
        self.global_counts[combo_name]['FP'] += np.logical_and(p_mask, ~g_mask).sum()
        self.global_counts[combo_name]['FN'] += np.logical_and(~p_mask, g_mask).sum()

    def add_case_metric(self, metric_dict):
        """
        Adds a single dictionary of results (e.g., {'Case': '001', 'Dice_1': 0.9}) 
        to the list of case-wise results.
        """
        self.case_metrics.append(metric_dict)
        
    def save_casewise_csv(self, filepath):
        """Saves the detailed per-case/per-slice metrics to CSV."""
        if not self.case_metrics: 
            return
        df = pd.DataFrame(self.case_metrics)
        df.to_csv(filepath, index=False)
        print(f"   Saved casewise metrics to: {filepath}")
        
    def save_global_summary_csv(self, filepath, extra_metrics_dict=None):
        """
        Calculates Global Dice (Total TP / (2*Total TP + FP + FN)) and saves summary.
        
        Args:
            filepath: output path for the CSV.
            extra_metrics_dict: Optional dictionary {Label: Value} to add extra columns 
                                (like 'Mean_HD95') to the summary.
        """
        rows = []
        for lbl, c in self.global_counts.items():
            denom = (2 * c['TP']) + c['FP'] + c['FN']
            # Global Dice formula
            dice = (2 * c['TP']) / denom if denom > 0 else 0.0
            
            entry = {'Label': lbl, 'Global_Dice': dice}
            
            # Append extra pre-calculated global stats (like Mean HD95)
            if extra_metrics_dict and lbl in extra_metrics_dict:
                entry['Global_Metric_Extra'] = extra_metrics_dict[lbl]
                
            rows.append(entry)
            
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            print(f"   Saved global summary to: {filepath}")