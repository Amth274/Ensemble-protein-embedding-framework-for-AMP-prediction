# GitHub Project Setup Guide

This guide explains how to set up the project board and issues in GitHub for tracking development progress.

## Step 1: Create Labels

Go to **Issues → Labels** and create the following labels:

| Label | Color | Description |
|-------|-------|-------------|
| `bug` | `#d73a4a` | Something isn't working |
| `enhancement` | `#a2eeef` | New feature or request |
| `testing` | `#0e8a16` | Related to testing infrastructure |
| `documentation` | `#0075ca` | Improvements or additions to documentation |
| `infrastructure` | `#fbca04` | Build, deploy, or infrastructure changes |
| `high-priority` | `#b60205` | Critical issues that should be addressed soon |
| `medium-priority` | `#d93f0b` | Important but not urgent |
| `low-priority` | `#fbca04` | Nice to have |
| `ml-ops` | `#5319e7` | Machine learning operations |
| `performance` | `#c5def5` | Performance improvements |
| `quick-fix` | `#006b75` | Small fixes that can be done quickly |
| `feature` | `#84b6eb` | New feature |
| `task` | `#d4c5f9` | Development task |
| `ci-cd` | `#1d76db` | Continuous integration and deployment |
| `models` | `#e99695` | Related to model architectures |
| `embeddings` | `#f9d0c4` | Related to ESM embeddings |
| `ensemble` | `#fef2c0` | Related to ensemble methods |
| `data` | `#c2e0c6` | Data processing and loading |
| `integration` | `#bfd4f2` | Integration tests |
| `jupyter` | `#d876e3` | Jupyter notebooks |
| `cli` | `#0e8a16` | Command-line interface |
| `api` | `#1d76db` | API documentation or development |
| `logging` | `#fbca04` | Logging functionality |
| `tracking` | `#5319e7` | Experiment tracking |
| `validation` | `#0075ca` | Input validation |
| `error-handling` | `#d73a4a` | Error handling |
| `optimization` | `#a2eeef` | Code or performance optimization |
| `deployment` | `#1d76db` | Deployment related |
| `docker` | `#0075ca` | Docker containerization |
| `cloud` | `#5319e7` | Cloud deployment |
| `community` | `#d876e3` | Community guidelines |
| `production` | `#b60205` | Production readiness |
| `getting-started` | `#0e8a16` | Getting started guides |

## Step 2: Create Milestones

Go to **Issues → Milestones** and create:

### Milestone 1: Testing Foundation
- **Due date**: 2 weeks from today
- **Description**: Establish comprehensive testing infrastructure with CI/CD
- **Issues**: #1, #2, #3, #4, #5, #6, #7

### Milestone 2: Documentation
- **Due date**: 3 weeks from today
- **Description**: Create comprehensive documentation and tutorials
- **Issues**: #8, #9, #10, #11

### Milestone 3: Production Ready
- **Due date**: 5 weeks from today
- **Description**: Add production features like logging, tracking, validation
- **Issues**: #12, #13, #14, #15, #16

### Milestone 4: Advanced Features
- **Due date**: No due date (ongoing)
- **Description**: Optional advanced features and optimizations
- **Issues**: #17, #18, #19, #20

### Milestone 5: Meta Tasks
- **Due date**: 1 week from today
- **Description**: Quick fixes and meta project setup
- **Issues**: #21, #22

## Step 3: Create Issues

Copy each issue from `docs/GITHUB_ISSUES.md` and create them in GitHub:

1. Go to **Issues → New Issue**
2. Choose the appropriate template (Bug Report, Feature Request, or Task)
3. Copy the content from `GITHUB_ISSUES.md`
4. Add the specified labels
5. Assign to the appropriate milestone
6. Assign team members (optional)

## Step 4: Create Project Board

### Option A: Classic Project Board

1. Go to **Projects → New Project → Classic**
2. Name it "AMP Prediction Development Roadmap"
3. Create columns:
   - 📋 **Backlog** - Issues not started
   - 🔄 **In Progress** - Currently working on
   - 👀 **In Review** - Pull request submitted
   - ✅ **Done** - Completed

### Option B: New Projects (Beta)

1. Go to **Projects → New Project → Table**
2. Name it "AMP Prediction Development Roadmap"
3. Add custom fields:
   - **Priority**: Single select (High, Medium, Low)
   - **Effort**: Single select (Small, Medium, Large)
   - **Category**: Single select (Testing, Docs, Feature, Bug)
4. Create views:
   - **By Priority**: Group by Priority field
   - **By Milestone**: Group by Milestone
   - **By Status**: Group by Status

## Step 5: Link Issues to Project

1. Go to each issue
2. Click "Projects" on the right sidebar
3. Add to your project board
4. Set initial status (usually "Backlog")

## Step 6: Set Up Automation (Optional)

### GitHub Actions Automation

Create `.github/workflows/project-automation.yml`:

```yaml
name: Project Automation
on:
  issues:
    types: [opened]
  pull_request:
    types: [opened, ready_for_review]

jobs:
  add-to-project:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/add-to-project@v0.4.0
        with:
          project-url: https://github.com/users/YOUR_USERNAME/projects/PROJECT_NUMBER
          github-token: ${{ secrets.ADD_TO_PROJECT_PAT }}
```

### Classic Project Automation

1. Go to your project board
2. Click "⋯" on each column
3. Set up automation:
   - **Backlog**: Newly added issues and PRs
   - **In Progress**: Reopened issues and PRs
   - **In Review**: Pull request opened or ready for review
   - **Done**: Issues closed, PRs merged

## Step 7: Create Issue Templates (Already Done!)

Issue templates are already created in `.github/ISSUE_TEMPLATE/`:
- `bug_report.md`
- `feature_request.md`
- `task.md`

## Step 8: Team Setup (If Collaborative)

1. **Assign roles**:
   - Project lead
   - Testing lead
   - Documentation lead
   - ML engineers

2. **Set up CODEOWNERS** (`.github/CODEOWNERS`):
```
# Testing
/amp_prediction/tests/ @testing-lead

# Documentation
/docs/ @docs-lead
*.md @docs-lead

# Core models
/amp_prediction/src/models/ @ml-lead

# Review everything
* @project-lead
```

3. **Set up branch protection**:
   - Require pull request reviews
   - Require status checks (tests) to pass
   - Require branches to be up to date

## Step 9: Weekly Workflow

### Sprint Planning (Start of Week)
1. Review backlog
2. Move items to "In Progress"
3. Assign issues to team members
4. Set weekly goals

### Daily Standups
1. What did you work on yesterday?
2. What will you work on today?
3. Any blockers?

### Sprint Review (End of Week)
1. Review completed items
2. Demo new features
3. Update project board
4. Plan next week

## Step 10: Pull Request Workflow

1. **Create feature branch**: `git checkout -b feature/issue-number-description`
2. **Make changes and commit**: Following conventional commits
3. **Push and create PR**: Link to issue with "Closes #issue-number"
4. **Request review**: Assign reviewers
5. **Address feedback**: Make changes as needed
6. **Merge**: Once approved and tests pass
7. **Delete branch**: Clean up after merge

## Example Conventional Commit Messages

```bash
# Features
git commit -m "feat: add SHAP-based model interpretability (#17)"

# Bug fixes
git commit -m "fix: correct ensemble voting threshold calculation (#4)"

# Documentation
git commit -m "docs: add tutorial notebook for custom datasets (#8)"

# Tests
git commit -m "test: add unit tests for CNN model architecture (#2)"

# Refactoring
git commit -m "refactor: simplify embedding generation pipeline"

# CI/CD
git commit -m "ci: add GitHub Actions workflow for automated testing (#7)"
```

## Tracking Progress

### View by Priority
```
High Priority:   ████████░░ 80% complete (8/10 issues)
Medium Priority: ████░░░░░░ 40% complete (4/10 issues)
Low Priority:    ██░░░░░░░░ 20% complete (2/10 issues)
```

### View by Milestone
```
Milestone 1 (Testing):      ██████░░░░ 60% complete
Milestone 2 (Docs):         ████░░░░░░ 40% complete
Milestone 3 (Production):   ░░░░░░░░░░  0% complete
Milestone 4 (Advanced):     ░░░░░░░░░░  0% complete
```

## Tips for Success

1. **Break large issues into smaller tasks** - Easier to track and complete
2. **Update issues regularly** - Comment on progress and blockers
3. **Use draft PRs** - For work in progress that needs early feedback
4. **Link related issues** - Use "Related to #issue-number" in descriptions
5. **Close issues with commits** - Use "Closes #issue-number" in PR description
6. **Review regularly** - Keep project board up to date
7. **Celebrate wins** - Acknowledge completed milestones

## Resources

- [GitHub Issues Documentation](https://docs.github.com/en/issues)
- [GitHub Projects Documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

*This setup guide was generated along with the repository restructuring on 2025-10-10*
