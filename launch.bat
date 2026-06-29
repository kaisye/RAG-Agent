@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND=%ROOT%backend"
set "FRONTEND=%ROOT%frontend"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

echo ============================================================
echo  RAG PDF Project - Launch
echo ============================================================
echo  Backend  : http://localhost:%BACKEND_PORT%
echo  Frontend : http://localhost:%FRONTEND_PORT%
echo  API docs : http://localhost:%BACKEND_PORT%/docs
echo ============================================================
echo.

:: --- Kill anything already on port 8000 ---
echo [0/2] Freeing port %BACKEND_PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%BACKEND_PORT% " ^| findstr "LISTENING"') do (
    echo       Killing PID %%P on port %BACKEND_PORT%
    taskkill /PID %%P /F >nul 2>&1
)

:: --- Kill anything already on port 5173 ---
echo [0/2] Freeing port %FRONTEND_PORT%...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%FRONTEND_PORT% " ^| findstr "LISTENING"') do (
    echo       Killing PID %%P on port %FRONTEND_PORT%
    taskkill /PID %%P /F >nul 2>&1
)

timeout /t 1 /nobreak >nul

:: --- Backend ---
echo [1/2] Starting backend (uvicorn on port %BACKEND_PORT%)...
start "RAG-Backend" cmd /k "cd /d "%BACKEND%" && uvicorn app.main:app --host 0.0.0.0 --port %BACKEND_PORT% --reload"

timeout /t 2 /nobreak >nul

:: --- Frontend ---
echo [2/2] Starting frontend (vite dev on port %FRONTEND_PORT%)...
start "RAG-Frontend" cmd /k "cd /d "%FRONTEND%" && npm run dev"

echo.
echo Both servers starting in separate windows.
echo Close those windows (or Ctrl+C inside each) to stop.
echo.
pause
