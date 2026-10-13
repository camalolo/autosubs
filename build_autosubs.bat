@echo off
rem Build autosubs.exe + whisper.dll (CUDA) from a fresh checkout.
rem Requires: CMake, Ninja, Visual Studio (MSVC), CUDA toolkit, git.

set ROOT=%~dp0

rem ---- third_party/whisper.cpp (cloned + patched) ----
if not exist "%ROOT%third_party\whisper.cpp\CMakeLists.txt" (
    echo [setup] cloning whisper.cpp v1.9.5 ...
    git clone --depth 1 --branch v1.9.5 https://github.com/ggml-org/whisper.cpp "%ROOT%third_party\whisper.cpp" || exit /b 1
)

rem apply the repetition-ratio patch if not already applied
git -C "%ROOT%third_party\whisper.cpp" apply --check "%ROOT%patches\whisper-cpp-repetition-ratio.patch" 2>NUL
if not errorlevel 1 (
    echo [setup] applying patches\whisper-cpp-repetition-ratio.patch ...
    git -C "%ROOT%third_party\whisper.cpp" apply "%ROOT%patches\whisper-cpp-repetition-ratio.patch" || exit /b 1
)

rem ---- locate Visual Studio ----
set VCVARS=
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>NUL`) do set VSDIR=%%i
if defined VSDIR set VCVARS=%VSDIR%\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" set VCVARS=E:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat

rem pinned toolset first (newest MSVC may be too new for nvcc), then unpinned
call "%VCVARS%" -vcvars_ver=14.44 >NUL 2>&1
cl 2>NUL | findstr /C:"19.44" >NUL
if errorlevel 1 (
    echo [setup] falling back to default MSVC toolset
    call "%VCVARS%"
    if errorlevel 1 exit /b 1
)

rem ---- build ----
cmake -S "%ROOT%." -B "%ROOT%build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="/O2 /fp:fast" || exit /b 1
cmake --build "%ROOT%build" --target autosubs whisper-cli -j 12 || exit /b 1

if not exist "%ROOT%bin" mkdir "%ROOT%bin"
copy /y "%ROOT%build\autosubs.exe" "%ROOT%bin\" >NUL
copy /y "%ROOT%build\bin\*.dll" "%ROOT%bin\" >NUL

echo BUILD_OK
