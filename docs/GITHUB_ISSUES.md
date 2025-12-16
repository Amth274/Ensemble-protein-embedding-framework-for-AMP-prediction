# GitHub Issues - Development Roadmap

This document contains all the issues that should be created in GitHub to track the development roadmap. Copy and paste each issue into GitHub.

---

## 🔴 HIGH PRIORITY - Phase 1: Testing & Quality Assurance

### Issue #1: Set Up Testing Infrastructure
**Labels**: `testing`, `infrastructure`, `high-priority`

#### Description
Set up comprehensive testing infrastructure using pytest to ensure code quality and catch regressions early.

#### Tasks
- [ ] Install testing dependencies: `pytest`, `pytest-cov`, `pytest-mock`
- [ ] Create `amp_prediction/pytest.ini` configuration file
- [ ] Create `amp_prediction/tests/conftest.py` with common fixtures
- [ ] Set up test data fixtures (mock embeddings, sample sequences)
- [ ] Configure coverage reporting
- [ ] Add pytest markers for slow tests, integration tests, etc.

#### Acceptance Criteria
- [ ] All testing dependencies are in `requirements.txt` under `[dev]`
- [ ] `pytest` runs successfully with `pytest tests/`
- [ ] Coverage report generates with `pytest --cov=src tests/`
- [ ] Test fixtures are reusable across test files

#### Implementation Notes
```bash
# pytest.ini example
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    unit: marks tests as unit tests
```

---

### Issue #2: Create Unit Tests for Model Architectures
**Labels**: `testing`, `models`, `high-priority`

#### Description
Write comprehensive unit tests for all neural network model architectures to ensure they work correctly with different input shapes and configurations.

#### Tasks
- [ ] Create `tests/test_models.py`
- [ ] Test `BaseAMPModel` functionality (save/load, get_info)
- [ ] Test `CNN1DAMPClassifier` forward pass and output shapes
- [ ] Test `AMPBilstmClassifier` forward pass and output shapes
- [ ] Test `GRUClassifier` forward pass and output shapes
- [ ] Test `CNN_BiLSTM_Classifier` (hybrid) forward pass
- [ ] Test `AMP_BiRNN` forward pass and output shapes
- [ ] Test `AMPTransformerClassifier` forward pass and output shapes
- [ ] Test `LogisticRegression` baseline model
- [ ] Test models with various batch sizes and sequence lengths
- [ ] Test model serialization (save/load state dict)

#### Acceptance Criteria
- [ ] All model architectures have at least 5 unit tests each
- [ ] Tests cover edge cases (batch_size=1, empty sequences, max length)
- [ ] Tests verify output shapes match expected dimensions
- [ ] Tests verify models can be saved and loaded correctly
- [ ] All tests pass with >80% code coverage for `src/models/`

#### Example Test
```python
def test_cnn_forward_pass():
    model = CNN1DAMPClassifier(embedding_dim=1280, seq_len=100)
    x = torch.randn(8, 100, 1280)  # batch=8, seq=100, dim=1280
    output = model(x)
    assert output.shape == (8,)  # Binary classification
```

---

### Issue #3: Create Unit Tests for Embedding Generation
**Labels**: `testing`, `embeddings`, `high-priority`

#### Description
Write tests for ESM embedding generation utilities, using mocked models to avoid requiring GPU/large downloads during testing.

#### Tasks
- [ ] Create `tests/test_embeddings.py`
- [ ] Mock ESM model to avoid downloading weights
- [ ] Test `ESMSequenceEmbedding` initialization
- [ ] Test sequence-level embedding generation (mean pooling)
- [ ] Test `ESMAminoAcidEmbedding` initialization
- [ ] Test per-residue embedding generation
- [ ] Test batch processing functionality
- [ ] Test padding/truncation to max_length
- [ ] Test save/load embedding functionality
- [ ] Test handling of invalid sequences

#### Acceptance Criteria
- [ ] Tests run without requiring GPU
- [ ] Tests run without downloading ESM model weights
- [ ] All embedding utilities have at least 4 unit tests
- [ ] Tests cover edge cases (very short/long sequences, special characters)
- [ ] All tests pass with >75% code coverage for `src/embeddings/`

---

### Issue #4: Create Unit Tests for Ensemble Methods
**Labels**: `testing`, `ensemble`, `high-priority`

#### Description
Write tests for ensemble voting strategies and ensemble classifier/regressor classes.

#### Tasks
- [ ] Create `tests/test_ensemble.py`
- [ ] Test `SoftVoting` strategy with various probabilities
- [ ] Test `HardVoting` strategy with various predictions
- [ ] Test `WeightedVoting` strategy with custom weights
- [ ] Test `EnsembleClassifier` initialization
- [ ] Test `EnsembleClassifier.predict_single_model()`
- [ ] Test `EnsembleClassifier.predict_all_models()`
- [ ] Test `EnsembleClassifier.predict_ensemble()`
- [ ] Test `EnsembleClassifier.evaluate_ensemble()`
- [ ] Test `EnsembleRegressor` functionality
- [ ] Test ensemble save/load functionality

#### Acceptance Criteria
- [ ] All ensemble strategies have at least 3 unit tests
- [ ] Ensemble classes have at least 5 unit tests
- [ ] Tests verify voting logic is correct
- [ ] Tests verify metrics calculation is accurate
- [ ] All tests pass with >80% code coverage for `src/ensemble/`

---

### Issue #5: Create Unit Tests for Data Processing
**Labels**: `testing`, `data`, `high-priority`

#### Description
Write tests for data loading, preprocessing, and dataset classes.

#### Tasks
- [ ] Create `tests/test_data.py`
- [ ] Test `AMPDataProcessor.load_data()` with CSV files
- [ ] Test sequence validation (valid amino acids only)
- [ ] Test data splitting functionality
- [ ] Test `SequenceDataset` initialization
- [ ] Test `SequenceDataset.__getitem__()` and `__len__()`
- [ ] Test padding/truncation in dataset
- [ ] Test handling of missing values
- [ ] Test handling of invalid file paths

#### Acceptance Criteria
- [ ] Data utilities have at least 4 unit tests each
- [ ] Tests use temporary files for I/O operations
- [ ] Tests verify data integrity after processing
- [ ] All tests pass with >75% code coverage for `src/data/`

---

### Issue #6: Create Integration Tests for Full Pipeline
**Labels**: `testing`, `integration`, `high-priority`

#### Description
Write end-to-end integration tests that verify the full training and prediction pipeline works correctly.

#### Tasks
- [ ] Create `tests/test_integration.py`
- [ ] Test full embedding generation workflow (with mock ESM)
- [ ] Test full training workflow (with small synthetic data)
- [ ] Test full prediction workflow
- [ ] Test ensemble evaluation workflow
- [ ] Test save/load entire ensemble workflow
- [ ] Test CLI commands (amp-embed, amp-train, amp-predict)
- [ ] Test Flask app endpoints
- [ ] Test error handling in full pipeline

#### Acceptance Criteria
- [ ] At least 5 integration tests covering major workflows
- [ ] Tests complete in reasonable time (<30 seconds each)
- [ ] Tests use temporary directories for outputs
- [ ] Tests clean up after themselves
- [ ] All tests pass consistently

---

### Issue #7: Set Up GitHub Actions CI/CD Pipeline
**Labels**: `ci-cd`, `infrastructure`, `high-priority`

#### Description
Set up automated testing and code quality checks using GitHub Actions.

#### Tasks
- [ ] Create `.github/workflows/tests.yml`
- [ ] Set up Python 3.8, 3.9, 3.10, 3.11 matrix testing
- [ ] Add automated pytest execution
- [ ] Add coverage reporting with codecov
- [ ] Create `.github/workflows/lint.yml`
- [ ] Add black code formatting check
- [ ] Add flake8 linting check
- [ ] Add mypy type checking
- [ ] Add badge to README.md for build status
- [ ] Add badge to README.md for coverage

#### Acceptance Criteria
- [ ] Tests run automatically on push and pull request
- [ ] Tests run on multiple Python versions
- [ ] Coverage reports upload to codecov.io
- [ ] Linting checks fail on style violations
- [ ] Badges display correctly in README

#### Example Workflow
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          cd amp_prediction
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          cd amp_prediction
          pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 🟡 MEDIUM PRIORITY - Phase 2: Documentation & Usability

### Issue #8: Create Tutorial Notebooks
**Labels**: `documentation`, `jupyter`, `medium-priority`

#### Description
Create comprehensive Jupyter notebooks demonstrating common workflows and use cases.

#### Tasks
- [ ] Create `02_Custom_Dataset_Tutorial.ipynb`
  - Loading custom CSV data
  - Generating embeddings for new sequences
  - Data quality checks
- [ ] Create `03_Model_Training_Guide.ipynb`
  - Training individual models
  - Hyperparameter tuning
  - Monitoring training progress
- [ ] Create `04_Ensemble_Evaluation.ipynb`
  - Evaluating individual models
  - Comparing model performance
  - Analyzing ensemble predictions
- [ ] Create `05_Production_Deployment.ipynb`
  - Loading trained models
  - Making predictions at scale
  - Deploying with Flask/FastAPI
- [ ] Add notebook requirements to `requirements.txt`
- [ ] Test all notebooks run without errors

#### Acceptance Criteria
- [ ] All notebooks run end-to-end without errors
- [ ] Notebooks include clear explanations and visualizations
- [ ] Notebooks use sample data (no large downloads required)
- [ ] Output cells show expected results
- [ ] Each notebook takes <5 minutes to run

---

### Issue #9: Enhance CLI Documentation
**Labels**: `documentation`, `cli`, `medium-priority`

#### Description
Improve command-line interface documentation with comprehensive help text and examples.

#### Tasks
- [ ] Add detailed help text to `amp-embed` command
- [ ] Add detailed help text to `amp-train` command
- [ ] Add detailed help text to `amp-evaluate` command
- [ ] Add detailed help text to `amp-predict` command
- [ ] Create `docs/CLI_GUIDE.md` with examples
- [ ] Add input validation with helpful error messages
- [ ] Add progress bars for long-running operations
- [ ] Add verbose/quiet modes

#### Acceptance Criteria
- [ ] `--help` flag shows clear, comprehensive information
- [ ] Examples provided for common use cases
- [ ] Error messages are helpful and actionable
- [ ] CLI follows standard conventions (--input, --output, etc.)

---

### Issue #10: Generate API Documentation
**Labels**: `documentation`, `api`, `medium-priority`

#### Description
Generate comprehensive API documentation from docstrings using Sphinx.

#### Tasks
- [ ] Install Sphinx and sphinx-rtd-theme
- [ ] Create `docs/conf.py` configuration
- [ ] Set up autodoc for all modules
- [ ] Document all public APIs with Google-style docstrings
- [ ] Add usage examples to docstrings
- [ ] Generate HTML documentation
- [ ] Host on Read the Docs or GitHub Pages
- [ ] Add link to README

#### Acceptance Criteria
- [ ] All public functions have docstrings
- [ ] Docstrings follow Google style guide
- [ ] API documentation builds without warnings
- [ ] Documentation is accessible online
- [ ] Examples in docstrings are tested

---

### Issue #11: Create Quick Start Guide
**Labels**: `documentation`, `getting-started`, `medium-priority`

#### Description
Create a comprehensive quick start guide for new users.

#### Tasks
- [ ] Create `docs/QUICKSTART.md`
- [ ] Installation instructions for different platforms
- [ ] First prediction in <5 minutes tutorial
- [ ] Common troubleshooting tips
- [ ] Links to detailed documentation
- [ ] Add to README.md
- [ ] Test instructions on fresh environment

#### Acceptance Criteria
- [ ] Guide is clear and concise
- [ ] New users can get started in <10 minutes
- [ ] Instructions work on Linux, macOS, Windows
- [ ] No assumed prior knowledge

---

## 🟢 LOWER PRIORITY - Phase 3: Production Features

### Issue #12: Implement Structured Logging
**Labels**: `logging`, `production`, `enhancement`

#### Description
Replace print statements with structured logging throughout the codebase.

#### Tasks
- [ ] Create logging configuration in `src/utils/logging_config.py`
- [ ] Replace print statements in `src/models/` with logger calls
- [ ] Replace print statements in `src/embeddings/` with logger calls
- [ ] Replace print statements in `src/ensemble/` with logger calls
- [ ] Replace print statements in `scripts/` with logger calls
- [ ] Add log levels (DEBUG, INFO, WARNING, ERROR)
- [ ] Add file logging to `logs/` directory
- [ ] Add JSON logging format option
- [ ] Document logging configuration in README

#### Acceptance Criteria
- [ ] No print statements remain in source code
- [ ] Logs include timestamp, level, module name
- [ ] Log verbosity is configurable
- [ ] Logs rotate to prevent disk space issues

---

### Issue #13: Add Experiment Tracking with Weights & Biases
**Labels**: `ml-ops`, `tracking`, `enhancement`

#### Description
Integrate Weights & Biases for experiment tracking and visualization.

#### Tasks
- [ ] Add `wandb` to optional dependencies
- [ ] Create `src/utils/experiment_tracking.py`
- [ ] Add wandb initialization to training scripts
- [ ] Log hyperparameters to wandb
- [ ] Log training/validation metrics to wandb
- [ ] Log model artifacts to wandb
- [ ] Add configuration for project/entity
- [ ] Add examples in documentation

#### Acceptance Criteria
- [ ] Experiments are logged to W&B automatically
- [ ] Metrics are visualized in W&B dashboard
- [ ] Hyperparameters are tracked
- [ ] Feature is optional (works without W&B)

---

### Issue #14: Implement Model Versioning
**Labels**: `ml-ops`, `versioning`, `enhancement`

#### Description
Add version tracking and metadata to saved models.

#### Tasks
- [ ] Add version field to model checkpoints
- [ ] Save training configuration with model
- [ ] Save performance metrics with model
- [ ] Save training timestamp and duration
- [ ] Create model registry system
- [ ] Add model comparison utilities
- [ ] Document model versioning scheme

#### Acceptance Criteria
- [ ] Models include version metadata
- [ ] Easy to compare models across versions
- [ ] Can load models by version
- [ ] Backwards compatibility maintained

---

### Issue #15: Add Input Validation and Error Handling
**Labels**: `validation`, `error-handling`, `enhancement`

#### Description
Add comprehensive input validation and meaningful error messages throughout the codebase.

#### Tasks
- [ ] Create `src/utils/validation.py` module
- [ ] Add sequence validation (amino acid composition)
- [ ] Add file path validation
- [ ] Add configuration validation
- [ ] Add model checkpoint validation
- [ ] Implement custom exception classes
- [ ] Add helpful error messages
- [ ] Add recovery suggestions in errors

#### Acceptance Criteria
- [ ] Invalid inputs are caught early with clear messages
- [ ] Errors suggest how to fix the issue
- [ ] No cryptic stack traces for user errors
- [ ] Edge cases are handled gracefully

---

### Issue #16: Optimize Embedding Generation Performance
**Labels**: `performance`, `optimization`, `enhancement`

#### Description
Improve embedding generation speed and memory efficiency.

#### Tasks
- [ ] Profile embedding generation bottlenecks
- [ ] Add multi-GPU support for batch processing
- [ ] Implement embedding caching mechanism
- [ ] Add progress bars with ETA
- [ ] Optimize batch size dynamically based on GPU memory
- [ ] Add resumption of interrupted generation
- [ ] Document performance benchmarks

#### Acceptance Criteria
- [ ] 2x speedup on large datasets
- [ ] Supports multiple GPUs
- [ ] Memory usage stays within limits
- [ ] Can resume interrupted operations

---

## 🔵 OPTIONAL - Phase 4: Advanced Features

### Issue #17: Add Hyperparameter Optimization with Optuna
**Labels**: `ml`, `optimization`, `feature`

#### Description
Integrate Optuna for automated hyperparameter tuning.

#### Tasks
- [ ] Add `optuna` to optional dependencies
- [ ] Create `scripts/hyperparameter_optimization.py`
- [ ] Define search space for each model
- [ ] Implement objective function
- [ ] Add pruning for early stopping
- [ ] Save best hyperparameters
- [ ] Create visualization of optimization
- [ ] Add example notebook

#### Acceptance Criteria
- [ ] Can optimize hyperparameters for all models
- [ ] Results are reproducible
- [ ] Optimization completes in reasonable time
- [ ] Documentation includes examples

---

### Issue #18: Implement Model Interpretability Tools
**Labels**: `ml`, `interpretability`, `feature`

#### Description
Add tools to interpret model predictions and visualize learned features.

#### Tasks
- [ ] Add attention weight visualization for Transformer
- [ ] Implement SHAP values for feature importance
- [ ] Create sequence motif analysis
- [ ] Add prediction confidence visualization
- [ ] Create interactive visualization dashboard
- [ ] Add example notebook demonstrating tools

#### Acceptance Criteria
- [ ] Can visualize why models make predictions
- [ ] Tools work for all model architectures
- [ ] Visualizations are publication-quality
- [ ] Documentation explains interpretation

---

### Issue #19: Create Docker Container for Deployment
**Labels**: `deployment`, `docker`, `feature`

#### Description
Create Docker container for easy deployment and reproducibility.

#### Tasks
- [ ] Create `Dockerfile` for CPU version
- [ ] Create `Dockerfile.gpu` for GPU version
- [ ] Create `docker-compose.yml`
- [ ] Add health check endpoint
- [ ] Optimize image size
- [ ] Push to Docker Hub
- [ ] Add deployment documentation
- [ ] Test on different platforms

#### Acceptance Criteria
- [ ] Docker image builds successfully
- [ ] Container runs Flask app
- [ ] Image size < 2GB
- [ ] Works on Linux, macOS, Windows Docker

---

### Issue #20: Deploy REST API to Cloud
**Labels**: `deployment`, `cloud`, `feature`

#### Description
Deploy the Flask REST API to a cloud platform with auto-scaling.

#### Tasks
- [ ] Choose cloud platform (AWS/GCP/Azure)
- [ ] Set up cloud infrastructure
- [ ] Configure auto-scaling
- [ ] Add load balancing
- [ ] Set up monitoring and alerting
- [ ] Configure CI/CD for automated deployment
- [ ] Add authentication and rate limiting
- [ ] Create deployment documentation

#### Acceptance Criteria
- [ ] API is accessible via public URL
- [ ] Auto-scales based on load
- [ ] <500ms latency for predictions
- [ ] 99.9% uptime
- [ ] Deployment is automated

---

## 📝 Meta Issues

### Issue #21: Fix .gitignore Duplicate Entry
**Labels**: `bug`, `quick-fix`

#### Description
Remove duplicate `*.pt` entry at the end of `.gitignore` file.

#### Tasks
- [ ] Open `.gitignore`
- [ ] Remove line 115: `*.pt` (duplicate)
- [ ] Verify other entries are correct

#### Acceptance Criteria
- [ ] No duplicate entries in `.gitignore`
- [ ] Sample embeddings are still excluded properly

---

### Issue #22: Create CONTRIBUTING.md
**Labels**: `documentation`, `community`

#### Description
Create contribution guidelines for external contributors.

#### Tasks
- [ ] Create `CONTRIBUTING.md`
- [ ] Code style guidelines
- [ ] Pull request process
- [ ] Testing requirements
- [ ] Documentation requirements
- [ ] Code of conduct
- [ ] How to report bugs
- [ ] How to suggest features

#### Acceptance Criteria
- [ ] Clear guidelines for contributors
- [ ] Follows community best practices
- [ ] Linked from README

---

## Priority Summary

**High Priority (Do First):**
- Issues #1-7: Testing infrastructure and CI/CD

**Medium Priority (Do Next):**
- Issues #8-11: Documentation and usability

**Lower Priority (Do Later):**
- Issues #12-16: Production features and optimization

**Optional (Nice to Have):**
- Issues #17-20: Advanced features and deployment

**Quick Fixes:**
- Issues #21-22: Meta tasks

---

## How to Use This Document

1. **Copy each issue** into GitHub Issues
2. **Assign labels** as specified
3. **Assign team members** to issues
4. **Create milestones** for each phase
5. **Track progress** using GitHub Projects
6. **Link issues** to pull requests when implementing

## Suggested Milestones

- **Milestone 1: Testing Foundation** (Issues #1-7) - Target: 2 weeks
- **Milestone 2: Documentation** (Issues #8-11) - Target: 1 week
- **Milestone 3: Production Ready** (Issues #12-16) - Target: 2 weeks
- **Milestone 4: Advanced Features** (Issues #17-20) - Target: Ongoing

---

*Generated: 2025-10-10*
*Last Updated: 2025-10-10*
