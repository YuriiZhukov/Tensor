@echo off
set "DIR=%~dp0/../build"

if exist "%DIR%" (
    rd /s /q "%DIR%"
)

mkdir "%DIR%"

cd "%DIR%"

cmake .. -DCMAKE_INSTALL_PREFIX=../install/ 
cmake --build . --target install --config Release
