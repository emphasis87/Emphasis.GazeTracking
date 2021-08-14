dotnet clean "src\Emphasis.GazeTracking.sln"
dotnet build -c Release "src\Emphasis.ComputerVision.sln"
cd "src\Emphasis.ComputerVision.Tests\"
dotnet run -c Release -- --filter *
cd ..\..
