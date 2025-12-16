# Documentation Directory

This directory contains all project documentation and development guides.

## 📁 Files Overview

### Development Guides
- **`GITHUB_ISSUES.md`** - 22 pre-written GitHub issues for the development roadmap
- **`PROJECT_SETUP.md`** - Complete guide for setting up GitHub project board and workflow
- **`RESTRUCTURING_SUMMARY.md`** - Detailed summary of repository restructuring
- **`RESTRUCTURING_CHANGES.md`** - Quick reference for restructuring changes

### Progress Documentation
- **`EXECUTION_RESULTS.md`** - Results from model training and evaluation
- **`IMPLEMENTATION_GUIDE.md`** - Implementation details and technical notes
- **`VALIDATION_FRAMEWORK_PROGRESS.md`** - Validation framework development progress
- **`training_log.txt`** - Training logs and metrics

## 🚀 Quick Start for Contributors

### 1. First Time Setup
```bash
# Read the main README
cat ../README.md

# Read restructuring changes
cat RESTRUCTURING_CHANGES.md

# Install dependencies
cd ../amp_prediction
pip install -e ".[dev]"
```

### 2. Set Up Development Environment
```bash
# Follow the project setup guide
cat PROJECT_SETUP.md

# Create GitHub issues from the template
# See GITHUB_ISSUES.md for all issues
```

### 3. Start Contributing
1. Pick an issue from the project board
2. Create a feature branch
3. Make changes and write tests
4. Submit a pull request
5. Get review and merge

## 📚 Documentation Map

### For Users
- **[Main README](../README.md)** - Project overview and usage
- **[CLAUDE.md](../CLAUDE.md)** - AI assistant guidance
- **Notebooks** (to be created) - Interactive tutorials

### For Contributors
- **[CONTRIBUTING.md](to be created)** - Contribution guidelines
- **[PROJECT_SETUP.md](PROJECT_SETUP.md)** - GitHub project setup
- **[GITHUB_ISSUES.md](GITHUB_ISSUES.md)** - Development roadmap

### For Developers
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Technical details
- **[EXECUTION_RESULTS.md](EXECUTION_RESULTS.md)** - Benchmark results
- **API Documentation** (to be generated) - Code reference

## 🎯 Development Phases

### Phase 1: Testing & Quality (High Priority)
Issues #1-7 focus on testing infrastructure and CI/CD
- **Timeline**: 2 weeks
- **Goal**: 80%+ test coverage, automated CI/CD

### Phase 2: Documentation (Medium Priority)
Issues #8-11 focus on tutorials and API docs
- **Timeline**: 1 week
- **Goal**: Complete user documentation

### Phase 3: Production Features (Medium Priority)
Issues #12-16 focus on logging, tracking, validation
- **Timeline**: 2 weeks
- **Goal**: Production-ready codebase

### Phase 4: Advanced Features (Optional)
Issues #17-20 focus on optimization and deployment
- **Timeline**: Ongoing
- **Goal**: Enhanced capabilities

## 📊 Issue Summary

| Priority | Count | Examples |
|----------|-------|----------|
| High | 7 | Testing infrastructure, CI/CD |
| Medium | 9 | Documentation, notebooks, logging |
| Low | 4 | Optimization, versioning |
| Optional | 4 | Hyperparameter tuning, deployment |

**Total: 22 issues** across 4 development phases

## 🔗 Quick Links

### Internal Documentation
- [Repository Structure](../CLAUDE.md#repository-structure)
- [Development Commands](../CLAUDE.md#development-commands)
- [Architecture Overview](../CLAUDE.md#architecture-overview)
- [Common Tasks](../CLAUDE.md#common-tasks)

### External Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [ESM Model](https://github.com/facebookresearch/esm)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [pytest Documentation](https://docs.pytest.org/)

## 📝 Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| GITHUB_ISSUES.md | ✅ Complete | 2025-10-10 |
| PROJECT_SETUP.md | ✅ Complete | 2025-10-10 |
| RESTRUCTURING_SUMMARY.md | ✅ Complete | 2025-10-10 |
| RESTRUCTURING_CHANGES.md | ✅ Complete | 2025-10-10 |
| EXECUTION_RESULTS.md | ✅ Complete | 2024-10-10 |
| IMPLEMENTATION_GUIDE.md | ✅ Complete | 2024-10-03 |
| VALIDATION_FRAMEWORK_PROGRESS.md | ✅ Complete | 2024-10-10 |

## 🤝 Getting Help

- **Questions?** Open an issue with the `question` label
- **Found a bug?** Use the bug report template
- **Want a feature?** Use the feature request template
- **Need clarification?** Check [CLAUDE.md](../CLAUDE.md) or ask in discussions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Note**: This documentation is maintained alongside the code. Please update relevant docs when making changes to the codebase.
