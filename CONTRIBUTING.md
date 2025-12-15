# Contributing to Stick-Gen

Thank you for your interest in contributing to Stick-Gen! We welcome contributions from the community to help us build the best open-source text-to-stick-figure animation model.

## getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/stick-gen.git
    cd stick-gen
    ```
3.  **Create a new branch** for your feature or bugfix:
    ```bash
    git checkout -b feature/my-awesome-feature
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Development Workflow

### Code Style

We follow standard Python coding conventions.
-   **Formatting**: We use `black` for code formatting.
-   **Linting**: We use `ruff` for linting.

Please run the following before committing:
```bash
pip install black ruff
black .
ruff check .
```

### Testing

Please ensure all tests pass before submitting your Pull Request.
```bash
python -m pytest tests/
```
If you are adding a new feature, please include appropriate unit and/or integration tests.

## Submitting a Pull Request

1.  Push your branch to your fork on GitHub.
2.  Open a Pull Request (PR) against the `main` branch of the `gestura-ai/stick-gen` repository.
3.  Fill out the PR template with a clear description of your changes.
4.  Link any relevant issues (e.g., "Fixes #123").

## Reporting Issues

If you find a bug or have a feature request, please use the [Issue Tracker](https://github.com/gestura-ai/stick-gen/issues).
-   Check existing issues to avoid duplicates.
-   Use the provided templates for Bug Reports and Feature Requests.
