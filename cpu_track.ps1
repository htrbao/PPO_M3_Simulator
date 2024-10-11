# Define the command you want to run (change the command here)
$command = "YourCommandHere.exe"

# Start the command as a background process
$process = Start-Process -FilePath $command -PassThru

# Initialize variables to track CPU usage
$cpuData = @()
$cpuPeak = 0
$totalSamples = 0

# Monitor the process CPU usage until the process exits
while (-not $process.HasExited) {
    # Get the CPU usage of the process
    $cpuUsage = (Get-Process -Id $process.Id | Select-Object -ExpandProperty CPU)

    # Calculate CPU usage per core (Optional)
    $cpuUsagePerCore = $cpuUsage / [Environment]::ProcessorCount

    # Store the CPU usage in an array
    $cpuData += $cpuUsagePerCore
    $totalSamples++

    # Track the peak CPU usage
    if ($cpuUsagePerCore -gt $cpuPeak) {
        $cpuPeak = $cpuUsagePerCore
    }

    # Wait for a second before next check
    Start-Sleep -Seconds 1
}

# Calculate the average CPU usage
$cpuAvg = ($cpuData | Measure-Object -Average).Average

# Output the results
Write-Host "Average CPU Usage: $cpuAvg %"
Write-Host "Peak CPU Usage: $cpuPeak %"

# Clean up
$process.Dispose()
