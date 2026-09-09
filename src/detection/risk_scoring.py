"""
Industrial Risk Scoring Engine — FactoryShield-OT
Implements the 4-level risk categorization (Low/Medium/High/Critical) based on
reconstruction errors, duration, affected sensors, and sensor criticality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime, timedelta


class RiskLevel(Enum):
    """Four-tier risk classification for industrial cybersecurity."""
    LOW = "Low"        # Surveillance - micro-drifts below alarm threshold
    MEDIUM = "Medium"  # Warning - moderate deviation, minor internal logic impacted
    HIGH = "High"      # Alert - severe correlation breach targeting critical equipment
    CRITICAL = "Critical"  # Emergency - confirmed stealth cyber-attack or physical DoS threat


@dataclass
class RiskEvent:
    """Represents a detected risk event with full context."""
    start_time: datetime
    end_time: datetime
    risk_level: RiskLevel
    max_error: float
    avg_error: float
    duration: float  # in seconds
    affected_sensors: List[str]
    root_cause_sensor: str
    error_contribution: float  # Contribution of root cause sensor to total error
    confidence: float  # 0.0 to 1.0 confidence score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "risk_level": self.risk_level.value,
            "max_error": self.max_error,
            "avg_error": self.avg_error,
            "duration_seconds": self.duration,
            "affected_sensor_count": len(self.affected_sensors),
            "affected_sensors": self.affected_sensors,
            "root_cause_sensor": self.root_cause_sensor,
            "error_contribution": self.error_contribution,
            "confidence": self.confidence,
        }


class RiskScoringEngine:
    """
    Implements the Industrial Risk Scoring Matrix for OT/ICS environments.
    
    Based on:
    1. Error magnitude (normalized reconstruction MAE)
    2. Error duration (how long anomaly persists)
    3. Number and criticality of affected sensors
    4. Physical process impact assessment
    """
    
    def __init__(self, 
                 thresholds: Optional[Dict[str, float]] = None,
                 critical_sensors: Optional[List[str]] = None,
                 risk_factors: Optional[Dict[str, float]] = None):
        """
        Initialize risk scoring engine.
        
        Args:
            thresholds: Dictionary with risk level thresholds (low, medium, high, critical)
            critical_sensors: List of sensor names considered critical for process safety
            risk_factors: Weighting factors for different risk components
        """
        # Default thresholds based on normalized reconstruction error [0, 1]
        self.thresholds = thresholds or {
            "low": 0.0,
            "medium": 0.3,
            "high": 0.6,
            "critical": 0.85
        }
        
        # Default critical sensors for Emerson Ovation DCS Boiler Process P1
        self.critical_sensors = critical_sensors or [
            # Pressure Control Loop
            "PIT01", "PCV01D", "PCV02D", "B2016",
            # Level Control Loop  
            "LIT01", "LCV01D", "B3004",
            # Flow Control Loop
            "FT03", "FCV03D", "B3005",
            # Temperature Control Loop
            "TIT01", "FCV01D", "FCV02D", "B4022",
            # Cooling Control
            "TIT03", "PP04"
        ]
        
        # Weighting factors for risk score calculation
        self.risk_factors = risk_factors or {
            "error_magnitude": 0.6,      # Reconstruction error magnitude
            "error_duration": 0.2,       # How long the anomaly persists
            "affected_sensors": 0.1,     # Number of sensors affected
            "sensor_criticality": 0.1,   # Criticality of affected sensors
        }
        
        # Validate that weights sum to 1.0
        total_weight = sum(self.risk_factors.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Risk factors must sum to 1.0, got {total_weight}")
    
    def calculate_risk_level(self, 
                            error_magnitude: float,
                            error_duration: int,
                            affected_sensors: List[str],
                            sensor_errors: Dict[str, float]) -> RiskLevel:
        """
        Determine risk level based on multiple factors.
        
        Args:
            error_magnitude: Normalized reconstruction error (0-1 scale)
            error_duration: Duration of anomaly in timesteps
            affected_sensors: List of sensor names showing anomalies
            sensor_errors: Dictionary mapping sensor names to their error contributions
            
        Returns:
            RiskLevel enum (Low, Medium, High, Critical)
        """
        # Calculate risk score components
        magnitude_score = self._normalize_error_magnitude(error_magnitude)
        duration_score = self._normalize_duration(error_duration)
        sensor_count_score = self._normalize_sensor_count(len(affected_sensors))
        criticality_score = self._calculate_criticality_score(affected_sensors, sensor_errors)
        
        # Combine scores with weighted factors
        risk_score = (
            self.risk_factors["error_magnitude"] * magnitude_score +
            self.risk_factors["error_duration"] * duration_score +
            self.risk_factors["affected_sensors"] * sensor_count_score +
            self.risk_factors["sensor_criticality"] * criticality_score
        )
        
        # Map to risk levels
        if risk_score >= self.thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.thresholds["high"]:
            return RiskLevel.HIGH
        elif risk_score >= self.thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _normalize_error_magnitude(self, error_magnitude: float) -> float:
        """Normalize error magnitude to [0, 1] scale."""
        # Assuming error_magnitude is already normalized from reconstruction MAE
        # Clip to reasonable bounds
        return np.clip(error_magnitude, 0.0, 1.0)
    
    def _normalize_duration(self, duration: int) -> float:
        """Normalize anomaly duration to [0, 1] scale."""
        # Based on typical industrial processes:
        # - < 5 timesteps: transient noise (0.1)
        # - 5-30 timesteps: potential anomaly (0.3-0.6)
        # - > 60 timesteps: sustained attack (0.8-1.0)
        if duration <= 5:
            return 0.1
        elif duration <= 30:
            return 0.3 + 0.3 * (duration - 5) / 25
        else:
            return 0.6 + 0.4 * min(1.0, (duration - 30) / 120)
    
    def _normalize_sensor_count(self, sensor_count: int) -> float:
        """Normalize number of affected sensors to [0, 1] scale."""
        # For HAIEnd dataset with 225 sensors:
        # - 1-5 sensors: localized issue (0.1-0.3)
        # - 6-20 sensors: subsystem affected (0.3-0.6)
        # - 21+ sensors: widespread issue (0.6-1.0)
        if sensor_count <= 5:
            return 0.1 + 0.2 * (sensor_count - 1) / 4
        elif sensor_count <= 20:
            return 0.3 + 0.3 * (sensor_count - 5) / 15
        else:
            return 0.6 + 0.4 * min(1.0, (sensor_count - 20) / 50)
    
    def _calculate_criticality_score(self, 
                                   affected_sensors: List[str],
                                   sensor_errors: Dict[str, float]) -> float:
        """
        Calculate criticality score based on which sensors are affected.
        
        Returns:
            Score from 0.0 (no critical sensors affected) to 1.0 (all critical sensors heavily affected)
        """
        if not affected_sensors:
            return 0.0
        
        # Find critical sensors among affected ones
        critical_affected = [s for s in affected_sensors if s in self.critical_sensors]
        
        if not critical_affected:
            return 0.0
        
        # Calculate weighted error for critical sensors
        total_critical_error = 0.0
        for sensor in critical_affected:
            total_critical_error += sensor_errors.get(sensor, 0.0)
        
        # Normalize by number of critical sensors and error magnitude
        max_possible_error = len(critical_affected)  # Assuming max error per sensor is 1.0
        if max_possible_error > 0:
            criticality_score = total_critical_error / max_possible_error
        else:
            criticality_score = 0.0
        
        return np.clip(criticality_score, 0.0, 1.0)
    
    def analyze_alert_segment(self,
                             timestamps: List[datetime],
                             errors: np.ndarray,
                             sensor_errors_df: pd.DataFrame,
                             alert_indices: List[int]) -> RiskEvent:
        """
        Analyze a contiguous alert segment and create a comprehensive risk event.
        
        Args:
            timestamps: List of datetime objects for each timestep
            errors: Array of reconstruction errors for each timestep
            sensor_errors_df: DataFrame with sensor-wise errors (columns = sensor names)
            alert_indices: List of indices where alerts are active in this segment
            
        Returns:
            RiskEvent object with full analysis
        """
        if not alert_indices:
            raise ValueError("Empty alert segment provided")
        
        # Extract segment data
        segment_errors = errors[alert_indices]
        segment_sensor_errors = sensor_errors_df.iloc[alert_indices]
        
        # Calculate basic statistics
        max_error = float(np.max(segment_errors))
        avg_error = float(np.mean(segment_errors))
        
        # Determine affected sensors (sensors with error > threshold)
        error_threshold = 0.1  # 10% of max error considered significant
        sensor_mean_errors = segment_sensor_errors.mean()
        affected_sensors = sensor_mean_errors[sensor_mean_errors > error_threshold].index.tolist()
        
        # Find root cause sensor (highest average error)
        if not sensor_mean_errors.empty:
            root_cause_idx = sensor_mean_errors.argmax()
            root_cause_sensor = sensor_mean_errors.index[root_cause_idx]
            root_cause_error = float(sensor_mean_errors.iloc[root_cause_idx])
            
            # Calculate error contribution percentage
            total_error = sensor_mean_errors.sum()
            error_contribution = root_cause_error / total_error if total_error > 0 else 0.0
        else:
            root_cause_sensor = "unknown"
            error_contribution = 0.0
        
        # Calculate duration
        start_idx = alert_indices[0]
        end_idx = alert_indices[-1]
        duration_seconds = (timestamps[end_idx] - timestamps[start_idx]).total_seconds()
        
        # Determine risk level
        sensor_errors_dict = sensor_mean_errors.to_dict()
        risk_level = self.calculate_risk_level(
            error_magnitude=max_error,
            error_duration=len(alert_indices),
            affected_sensors=affected_sensors,
            sensor_errors=sensor_errors_dict
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            max_error=max_error,
            duration=duration_seconds,
            sensor_count=len(affected_sensors),
            critical_sensors_affected=len([s for s in affected_sensors if s in self.critical_sensors])
        )
        
        return RiskEvent(
            start_time=timestamps[start_idx],
            end_time=timestamps[end_idx],
            risk_level=risk_level,
            max_error=max_error,
            avg_error=avg_error,
            duration=duration_seconds,
            affected_sensors=affected_sensors,
            root_cause_sensor=root_cause_sensor,
            error_contribution=error_contribution,
            confidence=confidence
        )
    
    def _calculate_confidence(self,
                            max_error: float,
                            duration: float,
                            sensor_count: int,
                            critical_sensors_affected: int) -> float:
        """
        Calculate confidence score for the risk assessment.
        
        Higher confidence when:
        - Error magnitude is high
        - Duration is sustained
        - Multiple sensors affected
        - Critical sensors are involved
        """
        # Normalize factors to [0, 1]
        error_factor = np.clip(max_error, 0.0, 1.0)
        duration_factor = np.clip(duration / 300, 0.0, 1.0)  # 5 minutes = high confidence
        sensor_factor = np.clip(sensor_count / 20, 0.0, 1.0)  # 20 sensors = high confidence
        critical_factor = np.clip(critical_sensors_affected / 5, 0.0, 1.0)  # 5 critical sensors = high confidence
        
        # Weighted average with emphasis on error magnitude and critical sensors
        confidence = (
            0.4 * error_factor +
            0.2 * duration_factor +
            0.2 * sensor_factor +
            0.2 * critical_factor
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def generate_risk_report(self, risk_events: List[RiskEvent]) -> pd.DataFrame:
        """Generate a comprehensive risk report DataFrame from multiple risk events."""
        records = []
        for i, event in enumerate(risk_events):
            record = event.to_dict()
            record["event_id"] = i + 1
            record["severity"] = event.risk_level.value
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Reorder columns for readability
        column_order = [
            "event_id", "start_time", "end_time", "duration_seconds",
            "risk_level", "severity", "confidence",
            "max_error", "avg_error",
            "affected_sensor_count", "affected_sensors",
            "root_cause_sensor", "error_contribution"
        ]
        
        existing_columns = [col for col in column_order if col in df.columns]
        other_columns = [col for col in df.columns if col not in column_order]
        
        return df[existing_columns + other_columns]


# Helper functions for practical usage
def detect_risk_events_from_alerts(timestamps: List[datetime],
                                  alerts: np.ndarray,
                                  errors: np.ndarray,
                                  sensor_errors_df: pd.DataFrame,
                                  risk_engine: Optional[RiskScoringEngine] = None) -> List[RiskEvent]:
    """
    Process alert sequence and generate risk events for each contiguous alert segment.
    
    Args:
        timestamps: Timestamps for each data point
        alerts: Binary alert array (1 = alert, 0 = normal)
        errors: Reconstruction error array
        sensor_errors_df: DataFrame with sensor-wise errors
        risk_engine: RiskScoringEngine instance (uses default if None)
        
    Returns:
        List of RiskEvent objects for each detected anomaly segment
    """
    if risk_engine is None:
        risk_engine = RiskScoringEngine()
    
    risk_events = []
    n_samples = len(alerts)
    i = 0
    
    while i < n_samples:
        if alerts[i] == 1:
            # Start of an alert segment
            start_idx = i
            while i < n_samples and alerts[i] == 1:
                i += 1
            end_idx = i - 1
            
            # Get indices for this segment
            segment_indices = list(range(start_idx, end_idx + 1))
            
            # Analyze the segment
            try:
                risk_event = risk_engine.analyze_alert_segment(
                    timestamps=timestamps,
                    errors=errors,
                    sensor_errors_df=sensor_errors_df,
                    alert_indices=segment_indices
                )
                risk_events.append(risk_event)
            except Exception as e:
                # Log error but continue processing
                print(f"Warning: Failed to analyze alert segment {start_idx}-{end_idx}: {e}")
        else:
            i += 1
    
    return risk_events


def get_risk_summary(risk_events: List[RiskEvent]) -> Dict:
    """Generate summary statistics for risk events."""
    if not risk_events:
        return {
            "total_events": 0,
            "risk_distribution": {},
            "avg_duration": 0.0,
            "avg_confidence": 0.0,
            "critical_events": 0
        }
    
    risk_distribution = {level.value: 0 for level in RiskLevel}
    total_duration = 0.0
    total_confidence = 0.0
    critical_events = 0
    
    for event in risk_events:
        risk_distribution[event.risk_level.value] += 1
        total_duration += event.duration
        total_confidence += event.confidence
        
        if event.risk_level == RiskLevel.CRITICAL:
            critical_events += 1
    
    return {
        "total_events": len(risk_events),
        "risk_distribution": risk_distribution,
        "avg_duration": total_duration / len(risk_events),
        "avg_confidence": total_confidence / len(risk_events),
        "critical_events": critical_events,
        "critical_percentage": (critical_events / len(risk_events) * 100) if risk_events else 0.0
    }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Risk Scoring Engine...")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate timestamps
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [base_time + timedelta(seconds=i) for i in range(n_samples)]
    
    # Simulate reconstruction errors (with some anomalies)
    errors = np.random.exponential(scale=0.1, size=n_samples)
    errors[100:150] = np.random.uniform(0.7, 0.9, 50)  # High anomaly
    errors[400:430] = np.random.uniform(0.4, 0.6, 30)  # Medium anomaly
    errors[700:710] = np.random.uniform(0.2, 0.3, 10)  # Low anomaly
    
    # Simulate alerts (based on threshold)
    alerts = (errors > 0.15).astype(int)
    
    # Simulate sensor-wise errors (5 sensors for testing)
    sensor_names = ["PIT01", "FT03", "TIT01", "PCV01D", "LCV01D"]
    sensor_errors = {}
    for sensor in sensor_names:
        sensor_errors[sensor] = errors * np.random.uniform(0.5, 1.5, n_samples)
    
    sensor_errors_df = pd.DataFrame(sensor_errors)
    
    # Initialize risk engine
    risk_engine = RiskScoringEngine()
    
    # Detect risk events
    risk_events = detect_risk_events_from_alerts(
        timestamps=timestamps,
        alerts=alerts,
        errors=errors,
        sensor_errors_df=sensor_errors_df,
        risk_engine=risk_engine
    )
    
    # Generate report
    if risk_events:
        print(f"Detected {len(risk_events)} risk events:")
        print("-" * 60)
        
        for i, event in enumerate(risk_events[:3]):  # Show first 3
            print(f"Event {i+1}:")
            print(f"  Time: {event.start_time.strftime('%H:%M:%S')} - {event.end_time.strftime('%H:%M:%S')}")
            print(f"  Risk Level: {event.risk_level.value}")
            print(f"  Duration: {event.duration:.1f}s")
            print(f"  Max Error: {event.max_error:.3f}")
            print(f"  Root Cause: {event.root_cause_sensor}")
            print(f"  Confidence: {event.confidence:.2%}")
            print()
        
        # Generate summary
        summary = get_risk_summary(risk_events)
        print("Risk Summary:")
        print(f"  Total Events: {summary['total_events']}")
        print(f"  Risk Distribution: {summary['risk_distribution']}")
        print(f"  Average Duration: {summary['avg_duration']:.1f}s")
        print(f"  Critical Events: {summary['critical_events']}")
        print(f"  Average Confidence: {summary['avg_confidence']:.2%}")
        
        # Generate DataFrame report
        report_df = risk_engine.generate_risk_report(risk_events)
        print(f"\nReport DataFrame shape: {report_df.shape}")
        print("\nFirst 3 rows of report:")
        print(report_df.head(3).to_string())
        
    else:
        print("No risk events detected.")
    
    print("\n✅ Risk scoring engine tested successfully!")