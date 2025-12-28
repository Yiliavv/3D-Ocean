<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version Change: N/A → 1.0.0 (Initial ratification)
Modified Principles: N/A (New constitution)
Added Sections:
  - Core Principles (5 principles)
  - Performance Standards
  - Development Workflow
  - Governance
Removed Sections: N/A

Templates Requiring Updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ Compatible (Success Criteria section exists)
  - .specify/templates/tasks-template.md: ✅ Compatible (Test phases exist)

Follow-up TODOs: None
================================================================================
-->

# Global Ocean Temperature Prediction Constitution

## Core Principles

### I. Modular Architecture

All code MUST follow modular design principles to ensure maintainability and testability:

- **Models**: MUST inherit from `LightningModule` base class
- **Datasets**: MUST inherit from `torch.utils.data.Dataset`
- **Training**: MUST use `BaseTrainer` unified interface; custom training loops are PROHIBITED
- **Components**: MUST be independently testable and replaceable
- **Dependencies**: MUST be explicitly declared; circular dependencies are PROHIBITED

**Rationale**: Ocean modeling involves complex pipelines. Modular design enables independent validation
of each component (data loading, model forward pass, loss computation) and simplifies debugging when
predictions deviate from expected values.

### II. Data Integrity & NaN Handling

All numerical computations MUST correctly handle NaN values representing land regions:

- **Loss Functions**: MUST mask NaN values before computing gradients
- **Metrics**: MUST exclude NaN regions from RMSE, MAE, and R² calculations
- **Visualization**: MUST render NaN regions distinctly (e.g., gray for land)
- **Data Pipeline**: MUST validate coordinate transformations preserve ocean/land masks
- **Temperature Values**: Values > 99°C or < -10°C MUST be treated as invalid

**Implementation Pattern**:
```python
def custom_mse_loss(self, y_pred, y):
    valid_mask = ~torch.isnan(y)
    if valid_mask.sum() > 0:
        return F.mse_loss(y_pred[valid_mask], y[valid_mask])
    return torch.tensor(0.0, requires_grad=True)
```

**Rationale**: Ignoring NaN handling corrupts loss gradients and produces physically impossible
predictions. This principle is NON-NEGOTIABLE for oceanographic applications.

### III. Reproducibility & Validation

All experiments MUST be reproducible and scientifically validated:

- **Random Seeds**: MUST use `set_seed(42)` or document chosen seed
- **Data Splits**: MUST use temporal ordering (no random shuffle for time series)
- **Checkpoints**: MUST save model state, optimizer state, and training configuration
- **Metrics Logging**: MUST record epoch-level loss, validation metrics, and final performance
- **Version Tracking**: MUST log PyTorch, Lightning, and key dependency versions

**Validation Requirements**:
- Every model MUST report RMSE on held-out validation set
- Predictions MUST be visually compared against ground truth maps
- Anomaly detection (SSTA) MUST be validated against historical NINO indices

**Rationale**: Climate science demands reproducibility. Results that cannot be reproduced have no
scientific value regardless of claimed performance.

### IV. Consistent User Experience

All interfaces (CLI, notebooks, visualization) MUST provide consistent, predictable behavior:

- **Coordinate Systems**: MUST use [-180, 180] longitude and [-90, 90] latitude consistently
- **Temperature Units**: MUST use Celsius for all user-facing outputs (convert from Kelvin internally)
- **Time Formats**: MUST use ISO 8601 (YYYY-MM-DD) for dates
- **Visualization**:
  - SST maps MUST use `jet` colormap with 0-30°C default range
  - Error maps MUST use `RdBu_r` colormap with symmetric range
  - All maps MUST include coastlines via Cartopy PlateCarree projection
- **Progress Feedback**: Long operations MUST display progress indicators
- **Error Messages**: MUST be actionable (include file paths, expected vs actual values)

**Rationale**: Researchers switch between notebooks, scripts, and visualizations frequently.
Inconsistent conventions waste time and introduce errors in interpretation.

### V. Performance & Efficiency

All implementations MUST meet computational and predictive performance standards:

- **Model Performance**: New SST models MUST achieve RMSE ≤ 0.5°C on global validation
- **Training Time**: Single epoch on full dataset SHOULD complete within 10 minutes on RTX 3090
- **Memory Usage**: Batch processing MUST NOT exceed 80% GPU memory
- **Data Loading**: DataLoader MUST use multi-worker loading (num_workers ≥ 4)
- **Vectorization**: NumPy operations MUST be vectorized; Python loops over arrays are PROHIBITED
- **Caching**: Repeated data access MUST use caching (e.g., annual file cache in ERA5 dataset)

**Rationale**: Ocean datasets are large (global 0.25° resolution = 1440×720 grid). Inefficient code
makes hyperparameter search and model iteration impractical.

## Performance Standards

### Model Accuracy Targets

| Model Type | Metric | Target | Acceptable |
|------------|--------|--------|------------|
| SST Prediction | RMSE | ≤ 0.40°C | ≤ 0.50°C |
| SST Prediction | R² | ≥ 0.95 | ≥ 0.90 |
| 3D Reconstruction | RMSE (0-200m) | ≤ 2.0°C | ≤ 2.5°C |
| NINO Index | Correlation | ≥ 0.90 | ≥ 0.85 |

### Computational Constraints

- **Inference Latency**: Single prediction MUST complete in < 1 second
- **Model Size**: Saved checkpoint SHOULD be < 500MB
- **Dataset Loading**: Full dataset initialization MUST complete in < 60 seconds

### Quality Gates

Before any model is considered production-ready:

1. ✅ Passes accuracy targets on validation set
2. ✅ Visualizations reviewed for physical plausibility
3. ✅ NaN handling verified on coastal regions
4. ✅ Temporal consistency checked (no sudden jumps between months)
5. ✅ Training logs archived with configuration

## Development Workflow

### Code Review Requirements

- All model changes MUST be reviewed for NaN handling correctness
- Performance-critical code MUST include benchmark results in PR description
- New visualizations MUST include sample outputs

### Testing Standards

- **Unit Tests**: Core utility functions (coordinate transforms, loss functions)
- **Integration Tests**: End-to-end training on small subset (first 12 months)
- **Regression Tests**: Saved predictions compared against baseline after changes

### Documentation Standards

- All classes MUST have docstrings with parameter descriptions
- Complex algorithms MUST include inline comments explaining oceanographic context
- Notebooks MUST have markdown cells explaining each analysis step

## Governance

This constitution supersedes all other development practices for this project.

### Amendment Process

1. Proposed changes MUST be documented with rationale
2. Changes affecting accuracy targets require validation on held-out data
3. All amendments MUST update the version number following semantic versioning:
   - MAJOR: Principle removal or redefinition
   - MINOR: New principle or section added
   - PATCH: Clarifications and typo fixes

### Compliance

- All pull requests MUST verify compliance with applicable principles
- Complexity beyond these guidelines MUST be justified in code comments
- Runtime development guidance available in `.cursor/rules/sst.mdc`

**Version**: 1.0.0 | **Ratified**: 2025-12-28 | **Last Amended**: 2025-12-28
