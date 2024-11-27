@echo off
setlocal enabledelayedexpansion

:: Define the directories to remove
set "dirs=__pycache__ .pytest_cache .vscode"

:: Loop over and remove the directories
for %%d in (%dirs%) do (
    for /d /r %%f in (%%d) do (
        echo Removing directory: %%f
        rmdir /s /q "%%f" 2>nul
    )
)

:: Collect all Python files
set "PYPATH="
for /r %%f in (*.py) do (
    set "PYPATH=!PYPATH! "%%f""
)

:: Check if there are any Python files to process
if not defined PYPATH (
    echo No Python files found.
    goto :EOF
)

:: Run Black to format the code and isort to sort imports
python -m black %PYPATH% && python -m isort %PYPATH%

endlocal
