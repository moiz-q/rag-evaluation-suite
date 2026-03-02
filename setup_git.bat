@echo off
echo Setting up Git repository for rag-evaluation-suite...
echo.

REM Initialize git if not already done
if not exist .git (
    echo Initializing Git repository...
    git init
    echo.
)

REM Add all files
echo Adding files to Git...
git add .
echo.

REM Commit
echo Committing files...
git commit -m "Initial commit: RAG Evaluation Suite - Automated metrics for measuring RAG quality"
echo.

REM Create GitHub repo and push
echo Creating GitHub repository and pushing...
gh repo create rag-evaluation-suite --public --source=. --description="Automated evaluation framework for RAG systems. Measure faithfulness, relevance, precision, and recall. Stop guessing, start measuring!" --push
echo.

echo Done! Repository created and pushed to GitHub.
echo.
pause
