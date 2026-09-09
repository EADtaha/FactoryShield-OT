"""
Configuration loader for FactoryShield-OT.
Loads and validates hyperparameters from YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

# Try to import yaml, but provide fallback if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    import json
    logging.warning("PyYAML not installed, using JSON fallback for configuration")

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset_name: str = "HAIEnd 23.05"
    target_system: str = "Emerson Ovation DCS (Boiler Process P1)"
    features_haiend: int = 225
    features_hai: int = 86
    scaler_type: str = "minmax"
    fit_on_normal_only: bool = True
    imputation_method: str = "ffill_bfill"
    window_size: int = 60
    stride: int = 1
    sequence_length: int = 60


@dataclass
class LSTMConfig:
    """LSTM Autoencoder configuration."""
    # FIX CQ-1: renamed hidden_size → hidden1 to match LSTMAutoencoder.__init__
    hidden1: int = 128
    latent_dim: int = 64  # FIX BUG-5: was 32, corrected to 64
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    loss_function: str = "l1"
    optimizer: str = "adam"
    seq_len: int = 60
    n_features: int = 225


@dataclass
class BaselineConfig:
    """Baseline models configuration."""
    isolation_forest: Dict[str, Any] = field(default_factory=lambda: {
        "contamination": 0.01,
        "n_estimators": 100,
        "max_samples": "auto",
        "random_state": 42
    })
    oneclass_svm: Dict[str, Any] = field(default_factory=lambda: {
        "nu": 0.1,
        "kernel": "rbf",
        "gamma": "scale",
        "random_state": 42
    })
    random_forest: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42
    })
    xgboost: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    })


@dataclass
class DetectionConfig:
    """Anomaly detection configuration."""
    error_metric: str = "mae"
    smoothing_method: str = "ema"
    ema_alpha: float = 0.1
    threshold_method: str = "percentile"
    percentile_threshold: float = 99.0
    min_alert_duration: int = 3
    suppress_alerts_after_detection: int = 60


@dataclass
class RiskScoringConfig:
    """Risk scoring configuration."""
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.0,
        "medium": 0.3,
        "high": 0.6,
        "critical": 0.85
    })
    factors: Dict[str, float] = field(default_factory=lambda: {
        "error_magnitude": 0.6,
        "error_duration": 0.2,
        "affected_sensors": 0.1,
        "sensor_criticality": 0.1
    })
    critical_sensors: list = field(default_factory=lambda: [
        "PIT01", "FT03", "TIT01", "PCV01D", "LCV01D", "FCV03D"
    ])


@dataclass
class XAIConfig:
    """Explainable AI configuration."""
    top_k_features: int = 5
    error_threshold: float = 0.1
    include_temporal_context: bool = True


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    clean_100: Dict[str, Any] = field(default_factory=lambda: {
        "description": "100% clean training data (normal operation only)",
        "contamination": 0.01,
        "use_supervised_models": False
    })
    mixed_70_30: Dict[str, Any] = field(default_factory=lambda: {
        "description": "70% clean, 30% anomalous training data",
        "contamination": 0.30,
        "use_supervised_models": True
    })


@dataclass
class FactoryShieldConfig:
    """Main configuration class for FactoryShield-OT."""
    
    # Core configurations
    data: DataConfig = field(default_factory=DataConfig)
    lstm_autoencoder: LSTMConfig = field(default_factory=LSTMConfig)
    baseline_models: BaselineConfig = field(default_factory=BaselineConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    risk_scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Path configurations
    paths: Dict[str, str] = field(default_factory=lambda: {
        "data_raw": "data/raw",
        "data_processed": "data/processed",
        "data_external": "data/external",
        "models": "models_saved",
        "logs": "logs",
        "notebooks": "notebooks",
        "app": "app",
        "configs": "configs",
        "train_clean": "data/processed/clean_100/train_clean_scaled.csv",
        "train_mixed": "data/processed/mixed_70_30/train_mixed_70_30.csv",
        "test_clean": "data/processed/clean_100/test_set_final.csv",
        "test_mixed": "data/processed/mixed_70_30/test_set_final.csv"
    })
    
    # Performance settings
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "max_ram_usage_gb": 8.0,
        "use_gpu": True,
        "gpu_memory_fraction": 0.8,
        "num_workers": 4,
        "prefetch_factor": 2,
        "inference_batch_size": 512,
        "evaluation_batch_size": 1024
    })
    
    # Logging settings
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "log_directory": "logs",
        "tensorboard": {
            "enabled": True,
            "log_dir": "logs/tensorboard",
            "update_freq": 100
        }
    })
    
    @classmethod
    def from_yaml(cls, config_path: str = "configs/hyperparameters.yaml") -> "FactoryShieldConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return cls()
        
        try:
            if YAML_AVAILABLE:
                with open(config_path, 'r') as f:
                    raw_config = yaml.safe_load(f)
            else:
                # Fallback to JSON if YAML not available
                with open(config_path, 'r') as f:
                    raw_config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.warning("Using default configuration")
            return cls()
        
        return cls.from_dict(raw_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FactoryShieldConfig":
        """Create configuration from dictionary."""
        # Extract nested configurations
        data_config = DataConfig(**config_dict.get("data", {}))
        lstm_config = LSTMConfig(**config_dict.get("lstm_autoencoder", {}))
        baseline_config = BaselineConfig(**config_dict.get("baseline_models", {}))
        detection_config = DetectionConfig(**config_dict.get("detection", {}))
        
        risk_scoring_dict = config_dict.get("risk_scoring", {})
        risk_config = RiskScoringConfig(
            thresholds=risk_scoring_dict.get("thresholds", {}),
            factors=risk_scoring_dict.get("factors", {}),
            critical_sensors=risk_scoring_dict.get("critical_sensors", [])
        )
        
        xai_dict = config_dict.get("xai", {}).get("root_cause", {})
        xai_config = XAIConfig(
            top_k_features=xai_dict.get("top_k_features", 5),
            error_threshold=xai_dict.get("error_threshold", 0.1),
            include_temporal_context=xai_dict.get("include_temporal_context", True)
        )
        
        experiment_config = ExperimentConfig(**config_dict.get("experiments", {}))
        
        # Create main configuration
        config = cls(
            data=data_config,
            lstm_autoencoder=lstm_config,
            baseline_models=baseline_config,
            detection=detection_config,
            risk_scoring=risk_config,
            xai=xai_config,
            experiments=experiment_config,
            paths=config_dict.get("paths", {}),
            performance=config_dict.get("performance", {}),
            logging=config_dict.get("logging", {})
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": {
                "dataset_name": self.data.dataset_name,
                "target_system": self.data.target_system,
                "features_haiend": self.data.features_haiend,
                "features_hai": self.data.features_hai,
                "scaler_type": self.data.scaler_type,
                "fit_on_normal_only": self.data.fit_on_normal_only,
                "imputation_method": self.data.imputation_method,
                "window_size": self.data.window_size,
                "stride": self.data.stride,
                "sequence_length": self.data.sequence_length,
            },
            "lstm_autoencoder": {
                "hidden1": self.lstm_autoencoder.hidden1,
                "latent_dim": self.lstm_autoencoder.latent_dim,
                "num_layers": self.lstm_autoencoder.num_layers,
                "dropout": self.lstm_autoencoder.dropout,
                "learning_rate": self.lstm_autoencoder.learning_rate,
                "batch_size": self.lstm_autoencoder.batch_size,
                "epochs": self.lstm_autoencoder.epochs,
                "loss_function": self.lstm_autoencoder.loss_function,
                "optimizer": self.lstm_autoencoder.optimizer,
                "seq_len": self.lstm_autoencoder.seq_len,
                "n_features": self.lstm_autoencoder.n_features,
            },
            "baseline_models": {
                "isolation_forest": self.baseline_models.isolation_forest,
                "oneclass_svm": self.baseline_models.oneclass_svm,
                "random_forest": self.baseline_models.random_forest,
                "xgboost": self.baseline_models.xgboost,
            },
            "detection": {
                "error_metric": self.detection.error_metric,
                "smoothing_method": self.detection.smoothing_method,
                "ema_alpha": self.detection.ema_alpha,
                "threshold_method": self.detection.threshold_method,
                "percentile_threshold": self.detection.percentile_threshold,
                "min_alert_duration": self.detection.min_alert_duration,
                "suppress_alerts_after_detection": self.detection.suppress_alerts_after_detection,
            },
            "risk_scoring": {
                "thresholds": self.risk_scoring.thresholds,
                "factors": self.risk_scoring.factors,
                "critical_sensors": self.risk_scoring.critical_sensors,
            },
            "xai": {
                "root_cause": {
                    "top_k_features": self.xai.top_k_features,
                    "error_threshold": self.xai.error_threshold,
                    "include_temporal_context": self.xai.include_temporal_context,
                }
            },
            "experiments": {
                "clean_100": self.experiments.clean_100,
                "mixed_70_30": self.experiments.mixed_70_30,
            },
            "paths": self.paths,
            "performance": self.performance,
            "logging": self.logging,
        }
    
    def save_yaml(self, config_path: str = "configs/hyperparameters.yaml"):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if YAML_AVAILABLE:
                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            else:
                # Fallback to JSON if YAML not available
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise


# Singleton instance for easy access
_config_instance: Optional[FactoryShieldConfig] = None


def get_config(config_path: str = "configs/hyperparameters.yaml") -> FactoryShieldConfig:
    """Get or create the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = FactoryShieldConfig.from_yaml(config_path)
    return _config_instance


def set_config(config: FactoryShieldConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Print some configuration values
    print("FactoryShield-OT Configuration")
    print("=" * 40)
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Target System: {config.data.target_system}")
    print(f"Window Size: {config.data.window_size} timesteps")
    print(f"LSTM Hidden Size: {config.lstm_autoencoder.hidden_size}")
    print(f"Risk Thresholds: {config.risk_scoring.thresholds}")
    print(f"Critical Sensors: {config.risk_scoring.critical_sensors[:3]}...")
    
    # Save configuration back (for verification)
    config.save_yaml("configs/hyperparameters_backup.yaml")
    print("\n✅ Configuration loaded and validated successfully!")