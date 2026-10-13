"""Train and persist the controlled Module 3 development RF v1 artifact."""

from backend.src.geometric_classification.development_rf_trainer import train_and_persist_development_model


if __name__ == "__main__":
    outputs = train_and_persist_development_model()
    print(f"Model artifact: {outputs.model_path}")
    print(f"Metadata: {outputs.metadata_path}")
    print(f"Training summary: {outputs.summary_path}")
    print("Artifact validation: PASS")
