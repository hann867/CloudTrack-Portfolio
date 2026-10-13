# CloudTrack command-line tools

Run these entry points as Python modules from the repository root so imports are
resolved consistently:

```powershell
python -m backend.cli.preprocessing.main_preprocessing
python -m backend.cli.dataset_preparation.main_dataset_audit
python -m backend.cli.geometric_classification.main_module3
```

The folders group command-line launchers by responsibility. Reusable application
code remains under `backend/src`, and the FastAPI application remains under
`backend/app`.
