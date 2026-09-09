"""
Metrics for FactoryShield-OT — implements eTaPR (enhanced time‑aware precision/recall) and
all anomaly‑detection metrics specified in the project plan.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)


class AnomalyDetectionMetrics:
    """Comprehensive metrics for industrial anomaly detection."""

    @staticmethod
    def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> dict:
        """Compute standard binary classification metrics.
        
        Args:
            y_true: Ground truth labels (0=normal, 1=anomaly).
            y_pred: Binary predictions (0/1).
            y_scores: Continuous anomaly scores (optional for AUC).
            
        Returns:
            Dictionary with accuracy, precision, recall, F1, FPR, FNR, confusion matrix.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        
        if y_scores is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
            fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            metrics["pr_auc"] = average_precision_score(y_true, y_scores)
            metrics["roc_curve"] = (fpr_curve, tpr_curve)
            metrics["pr_curve"] = (precision_curve, recall_curve)
        
        return metrics

    @staticmethod
    def etapr_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                      theta_p: float = 0.5, theta_r: float = 0.5) -> dict:
        """Enhanced Time‑Aware Precision/Recall (eTaPR).
        
        Implementation follows the HAICon 2021 evaluation methodology:
        - Precision is adjusted based on the detection overlap with ground‑truth anomalies.
        - Recall considers both detection presence and timely coverage.
        
        Args:
            y_true: Ground truth anomaly labels.
            y_pred: Binary predictions.
            theta_p: Minimum overlap ratio for a detection to be considered correct (0.5 = 50%).
            theta_r: Minimum coverage ratio for an anomaly to be considered detected (0.5 = 50%).
            
        Returns:
            Dictionary with eTaP (time‑aware precision), eTaR (time‑aware recall), eTaF1.
        """
        # Find contiguous anomaly segments in ground truth and predictions
        true_segments = AnomalyDetectionMetrics._find_segments(y_true)
        pred_segments = AnomalyDetectionMetrics._find_segments(y_pred)
        
        if not true_segments and not pred_segments:
            return {"eTaP": 1.0, "eTaR": 1.0, "eTaF1": 1.0}
        
        # Compute eTaP
        correct_detections = 0
        total_predicted = len(pred_segments)
        
        for p_start, p_end in pred_segments:
            detected = False
            for t_start, t_end in true_segments:
                overlap_start = max(p_start, t_start)
                overlap_end = min(p_end, t_end)
                if overlap_start < overlap_end:  # Non‑zero overlap
                    overlap_duration = overlap_end - overlap_start
                    pred_duration = p_end - p_start
                    true_duration = t_end - t_start
                    
                    # Check if overlap ratio meets theta_p threshold
                    if overlap_duration / pred_duration >= theta_p:
                        detected = True
                        break
            if detected:
                correct_detections += 1
        
        eTaP = correct_detections / total_predicted if total_predicted > 0 else 0.0
        
        # Compute eTaR
        detected_anomalies = 0
        total_true = len(true_segments)
        
        for t_start, t_end in true_segments:
            covered = False
            for p_start, p_end in pred_segments:
                overlap_start = max(p_start, t_start)
                overlap_end = min(p_end, t_end)
                if overlap_start < overlap_end:  # Non‑zero overlap
                    overlap_duration = overlap_end - overlap_start
                    true_duration = t_end - t_start
                    
                    # Check if coverage ratio meets theta_r threshold
                    if overlap_duration / true_duration >= theta_r:
                        covered = True
                        break
            if covered:
                detected_anomalies += 1
        
        eTaR = detected_anomalies / total_true if total_true > 0 else 0.0
        
        # Compute eTaF1
        eTaF1 = 2 * eTaP * eTaR / (eTaP + eTaR) if (eTaP + eTaR) > 0 else 0.0
        
        return {
            "eTaP": eTaP,
            "eTaR": eTaR,
            "eTaF1": eTaF1,
            "theta_p": theta_p,
            "theta_r": theta_r,
            "true_segments": len(true_segments),
            "pred_segments": len(pred_segments),
            "correct_detections": correct_detections,
            "detected_anomalies": detected_anomalies,
        }

    @staticmethod
    def _find_segments(labels: np.ndarray) -> list:
        """Extract contiguous segments where label == 1."""
        segments = []
        n = len(labels)
        i = 0
        while i < n:
            if labels[i] == 1:
                start = i
                while i < n and labels[i] == 1:
                    i += 1
                segments.append((start, i))
            else:
                i += 1
        return segments

    @staticmethod
    def detection_latency(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute detection latency (time from anomaly start to first detection).
        
        Args:
            y_true: Ground truth anomaly labels.
            y_pred: Binary predictions.
            
        Returns:
            Dictionary with mean/median latency, latency distribution.
        """
        true_segments = AnomalyDetectionMetrics._find_segments(y_true)
        latencies = []
        
        for t_start, t_end in true_segments:
            # Find first detection within or after this anomaly
            first_detection = None
            for i in range(t_start, min(t_end + 100, len(y_pred))):  # Look 100 steps ahead
                if i < len(y_pred) and y_pred[i] == 1:
                    first_detection = i
                    break
            
            if first_detection is not None:
                latency = max(0, first_detection - t_start)
                latencies.append(latency)
        
        if not latencies:
            return {
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "latencies": [],
                "detected_segments": 0,
                "total_segments": len(true_segments),
            }
        
        return {
            "mean_latency": float(np.mean(latencies)),
            "median_latency": float(np.median(latencies)),
            "min_latency": float(np.min(latencies)),
            "max_latency": float(np.max(latencies)),
            "latencies": latencies,
            "detected_segments": len(latencies),
            "total_segments": len(true_segments),
        }

    @staticmethod
    def comprehensive_report(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_scores: np.ndarray = None, model_name: str = "") -> str:
        """Generate a comprehensive textual report.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Binary predictions.
            y_scores: Anomaly scores (optional).
            model_name: Name of the model being evaluated.
            
        Returns:
            Formatted string report.
        """
        basic = AnomalyDetectionMetrics.basic_metrics(y_true, y_pred, y_scores)
        etapr = AnomalyDetectionMetrics.etapr_metrics(y_true, y_pred)
        latency = AnomalyDetectionMetrics.detection_latency(y_true, y_pred)
        
        report = f"{'='*60}\n"
        report += f"📊 COMPREHENSIVE EVALUATION REPORT - {model_name}\n"
        report += f"{'='*60}\n\n"
        
        report += "🔹 BASIC CLASSIFICATION METRICS:\n"
        report += f"  • Accuracy:            {basic['accuracy']:.4f}\n"
        report += f"  • Precision:           {basic['precision']:.4f}\n"
        report += f"  • Recall (TPR):        {basic['recall']:.4f}\n"
        report += f"  • F1-Score:            {basic['f1_score']:.4f}\n"
        report += f"  • FPR (False Positive): {basic['false_positive_rate']:.4f}\n"
        report += f"  • FNR (False Negative): {basic['false_negative_rate']:.4f}\n"
        
        if 'roc_auc' in basic:
            report += f"  • ROC-AUC:             {basic['roc_auc']:.4f}\n"
            report += f"  • PR-AUC:              {basic['pr_auc']:.4f}\n"
        
        report += f"\n🔹 CONFUSION MATRIX:\n"
        report += f"    [TN: {basic['true_negatives']:<6} | FP: {basic['false_positives']}]\n"
        report += f"    [FN: {basic['false_negatives']:<6} | TP: {basic['true_positives']}]\n"
        
        report += f"\n🔹 ENHANCED TIME-AWARE METRICS (eTaPR):\n"
        report += f"  • eTaP (θ={etapr['theta_p']}):    {etapr['eTaP']:.4f}\n"
        report += f"  • eTaR (θ={etapr['theta_r']}):    {etapr['eTaR']:.4f}\n"
        report += f"  • eTaF1:                {etapr['eTaF1']:.4f}\n"
        report += f"  • True anomaly segments: {etapr['true_segments']}\n"
        report += f"  • Detected segments:     {etapr['detected_anomalies']}/{etapr['true_segments']}\n"
        
        report += f"\n🔹 DETECTION LATENCY (timesteps):\n"
        report += f"  • Mean latency:         {latency['mean_latency']:.2f}\n"
        report += f"  • Median latency:       {latency['median_latency']:.2f}\n"
        report += f"  • Range:                [{latency['min_latency']:.0f}, {latency['max_latency']:.0f}]\n"
        report += f"  • Detected segments:    {latency['detected_segments']}/{latency['total_segments']}\n"
        
        report += f"\n{'='*60}\n"
        
        return report


# Convenience function for quick evaluation
def evaluate_anomaly_detection(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_scores: np.ndarray = None, model_name: str = "") -> dict:
    """One-call evaluation returning all metrics."""
    return {
        "basic": AnomalyDetectionMetrics.basic_metrics(y_true, y_pred, y_scores),
        "etapr": AnomalyDetectionMetrics.etapr_metrics(y_true, y_pred),
        "latency": AnomalyDetectionMetrics.detection_latency(y_true, y_pred),
        "report": AnomalyDetectionMetrics.comprehensive_report(y_true, y_pred, y_scores, model_name),
    }


# Test the implementation
if __name__ == "__main__":
    # Generate synthetic test data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.zeros(n_samples)
    y_pred = np.zeros(n_samples)
    y_scores = np.random.rand(n_samples)
    
    # Add some anomalies
    y_true[100:150] = 1
    y_true[400:450] = 1
    y_true[700:730] = 1
    
    # Predictions with some errors
    y_pred[110:140] = 1  # Good detection (slightly delayed)
    y_pred[410:430] = 1  # Partial detection
    y_pred[600:610] = 1  # False positive
    # y_pred[700:730] = 0  # Missed entirely
    
    # Test metrics
    metrics = evaluate_anomaly_detection(y_true, y_pred, y_scores, "Test Model")
    print(metrics["report"])
    
    # Quick verification
    print(f"\n✅ Basic metrics keys: {list(metrics['basic'].keys())}")
    print(f"✅ eTaPR metrics keys: {list(metrics['etapr'].keys())}")
    print(f"✅ Latency metrics keys: {list(metrics['latency'].keys())}")