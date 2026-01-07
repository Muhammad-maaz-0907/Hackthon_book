import subprocess
import sys

def install_package(package):
    """Install a single package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False

def main():
    packages = [
        "python-dotenv",
        "pydantic",
        "fastapi",
        "uvicorn",
        "qdrant-client",
        "asyncpg",
        "openai",
        "tiktoken",
        "sentence-transformers",
        "PyYAML",
        "python-multipart",
        "beautifulsoup4",
        "markdown"
    ]

    print("Installing backend dependencies...")

    for package in packages:
        print(f"Installing {package}...")
        success = install_package(package)
        if not success:
            print(f"Failed to install {package}, continuing with others...")

    print("Package installation completed.")

if __name__ == "__main__":
    main()