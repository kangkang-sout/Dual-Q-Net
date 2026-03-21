@echo off
echo ========================================
echo Dual Q-Net Java Implementation
echo ========================================

echo.
echo Checking Java version...
java -version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Java not found. Please install Java 11 or higher.
    pause
    exit /b 1
)

echo.
echo Checking Maven...
mvn -version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Maven not found. Please install Maven 3.6+.
    pause
    exit /b 1
)

echo.
echo Compiling project...
mvn clean compile
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Compilation failed.
    pause
    exit /b 1
)

echo.
echo Running Dual Q-Net demonstration...
mvn exec:java -Dexec.mainClass="com.cognitivediagnosis.Main"

echo.
echo ========================================
echo Execution completed!
echo ========================================
pause
