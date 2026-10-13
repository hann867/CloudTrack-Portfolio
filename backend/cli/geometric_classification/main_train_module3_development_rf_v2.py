"""Train and persist the controlled Module 3 Development RF v2 artifact."""

from backend.src.geometric_classification.final_assignment_rf_trainer import (
    train_and_persist_final_assignment_model,
)


if __name__ == "__main__":
    outputs = train_and_persist_final_assignment_model()
    print(f"Model artifact: {outputs.model_path}")
    print(f"Metadata: {outputs.metadata_path}")
    print(f"Training summary: {outputs.summary_path}")
    print("Artifact reload validation: PASS")
