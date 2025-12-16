# GitHub Actions Test Fixes

## Problem
GitHub Actions CI/CD pipeline was failing with Ruff linting errors:
- 14 undefined name errors in `templates/` directory
- Files: `templates/phase1_schema_extensions.py`, `templates/phase1_training_loop.py`
- Errors: Undefined names like `ActionType`, `ActorType`, `Position`, `StickFigureTransformer`, etc.

## Root Cause
The `templates/` directory contains **code snippets and examples** meant to be integrated into other files, not standalone Python modules. These files reference types and functions that are defined elsewhere in the codebase.

## Solution

### 1. Created `ruff.toml` Configuration
Added a Ruff configuration file to exclude template and data directories from linting:

```toml
# Exclude template files and other non-executable code
exclude = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "*.egg-info",
    "build",
    "dist",
    "templates",  # Template files contain code snippets, not standalone modules
    "data",       # Third-party datasets and preprocessing code
]
```

### 2. Updated GitHub Actions Workflow
Modified `.github/workflows/tests.yml` to exclude templates and data directories:

**Before:**
```yaml
- name: Lint with Ruff
  run: |
    ruff check . --select=E9,F63,F7,F82 --target-version=py310
    ruff check . --exit-zero

- name: Check formatting with Black
  run: |
    black . --check
```

**After:**
```yaml
- name: Lint with Ruff
  run: |
    # exclude templates/ directory (contains code snippets, not standalone modules)
    ruff check . --select=E9,F63,F7,F82 --target-version=py310 --exclude=templates
    ruff check . --exit-zero --exclude=templates

- name: Check formatting with Black
  run: |
    black . --check --exclude="(templates|data)"
```

### 3. Formatted Codebase
Ran Black formatter on the entire codebase (excluding templates and data):
- Reformatted 67 files
- All files now pass Black formatting checks

## Verification

### Local Testing
```bash
# Ruff linting (strict mode)
$ ruff check . --select=E9,F63,F7,F82 --target-version=py310
All checks passed!

# Black formatting
$ black . --check --exclude="(templates|data)"
All done! ✨ 🍰 ✨
78 files would be left unchanged.
```

### Files Modified
1. **Created:** `ruff.toml` - Ruff configuration with exclusions
2. **Modified:** `.github/workflows/tests.yml` - Updated linting commands
3. **Formatted:** 67 Python files across `src/`, `scripts/`, `examples/`, `tests/`

## Impact
✅ **GitHub Actions CI/CD will now pass** the linting and formatting checks
✅ **Template files are properly excluded** from strict linting
✅ **Third-party data code is excluded** from formatting requirements
✅ **Codebase is consistently formatted** with Black

## Next Steps
When the next commit is pushed to GitHub, the Actions workflow will:
1. ✅ Pass Ruff linting (no undefined names in templates)
2. ✅ Pass Black formatting (all code properly formatted)
3. Continue to pytest tests (may have separate import issues to address)

## Notes
- Template files in `templates/` are intentionally incomplete code snippets
- They are meant to be copied/integrated into actual source files
- Excluding them from linting is the correct approach
- The `data/` directory contains third-party code (HumanML3D, etc.) that we don't control

