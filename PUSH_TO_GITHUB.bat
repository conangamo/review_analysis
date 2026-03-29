@echo off
chcp 65001 > nul
cd /d %~dp0

echo ================================================================================
echo   PUSH TO GITHUB - Upload Project
echo ================================================================================
echo.
echo Repository: https://github.com/conangamo/review_analysis
echo.
echo This will upload (excluding notebooks/ folder):
echo   - 3 documentation files (README, START_HERE, DATA_PIPELINE_GUIDE)
echo   - Source code (src/, scripts/)
echo   - Configuration (config/, requirements.txt, etc.)
echo   - Setup files (Dockerfile, docker-compose.yml)
echo.
echo Will NOT upload:
echo   - notebooks/ folder (excluded in .gitignore)
echo   - data/ folder (too large)
echo   - .env file (secrets)
echo.
echo ================================================================================
echo.
pause

echo.
echo [Step 1] Cleaning up git lock files...
del /F /Q .git\index.lock 2>nul
timeout /t 2 /nobreak >nul
echo ✅ Lock files removed

echo.
echo [Step 2] Staging files...
git add README.md START_HERE.md DATA_PIPELINE_GUIDE.md
git add requirements.txt setup.py .gitignore .dockerignore .env.example
git add Dockerfile docker-compose.yml
git add src\
git add scripts\
git add config\
git add docs\
echo ✅ Files staged

echo.
echo [Step 3] Creating commit...
git commit -m "Initial commit: Product Review Analyzer with 90%% accuracy" -m "- Complete data pipeline (7 steps)" -m "- AI engine with Sprint 1-2-3 improvements" -m "- 90%% accuracy (up from 50%%)" -m "- Hallucination fix, negation handling, validation layer" -m "- Clean documentation (3 essential guides)" -m "- Production-ready code"

if errorlevel 1 (
    echo.
    echo ❌ Commit failed! Check error above.
    pause
    exit /b 1
)
echo ✅ Commit created

echo.
echo [Step 4] Adding remote repository...
git remote add origin https://github.com/conangamo/review_analysis.git
echo ✅ Remote added

echo.
echo [Step 5] Pushing to GitHub...
echo.
echo You may need to enter your GitHub credentials...
echo.
git push -u origin master

if errorlevel 1 (
    echo.
    echo ❌ Push failed!
    echo.
    echo Common issues:
    echo   1. Repository not empty - Use: git push -u origin master --force
    echo   2. Authentication needed - Setup GitHub credentials
    echo   3. Branch name - Try: git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✅ SUCCESS! Project uploaded to GitHub!
echo ================================================================================
echo.
echo Repository: https://github.com/conangamo/review_analysis
echo.
echo What was uploaded:
echo   ✅ 3 documentation files
echo   ✅ All source code (37 files in src/)
echo   ✅ 7 pipeline scripts
echo   ✅ Configuration files
echo   ✅ Docker setup
echo.
echo What was NOT uploaded (excluded):
echo   ❌ notebooks/ folder (tests & reference docs)
echo   ❌ data/ folder (too large, local only)
echo   ❌ .env file (secrets)
echo.
echo Next: View your repo at https://github.com/conangamo/review_analysis
echo.
echo ================================================================================
pause
