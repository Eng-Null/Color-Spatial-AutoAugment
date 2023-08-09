# Check if the virtual environment exit else create one
if (Test-Path .venv) {
    Write-Output ".venv directory exists."
} else {
    Write-Output "Creating .venv directory..."
    python -m venv .venv
}


# Activate virtual environment
. .\.venv\Scripts\Activate.ps1
Write-Output "Activating virtual environment..."

# Check if Tensorboard is installed else install it
function Check-AndInstall-Package {
    param (
        [string]$packageName
    )

    $output = & python -m pip show $packageName 2>&1

    if ($output -match "Name\s*:\s*$packageName") {
        Write-Output "$packageName is already installed."
    } else {
        Write-Output "$packageName is not installed. Installing..."
        try {
            # Use the `pip` module in PowerShell to install the package
            & python -m pip install $packageName
        } catch {
            Write-Output "Failed to install $packageName. Please make sure you have Python and 'pip' installed."
            return
        }
        Write-Output "$packageName has been successfully installed."
    }
}

$packageNameToCheck = "tensorboard"
Check-AndInstall-Package -packageName $packageNameToCheck

# Run TensorBoard
tensorboard --logdir='F:\TESTS\Python AutoAugment SimCLR' --host localhost --port 8088