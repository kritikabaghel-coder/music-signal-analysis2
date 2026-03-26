"""
Verify project setup and dependencies.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or version.minor < 8:
        print("  ✗ Python 3.8+ required")
        return False
    print("  ✓ Python version OK")
    return True


def check_imports():
    """Check required library imports."""
    required = {
        "librosa": "librosa",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "soundfile": "soundfile"
    }

    print("\nChecking imports:")
    all_ok = True
    for lib_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✓ {lib_name}")
        except ImportError:
            print(f"  ✗ {lib_name} (install: pip install {lib_name})")
            all_ok = False

    return all_ok


def check_directory_structure():
    """Check project directory structure."""
    print("\nChecking directory structure:")
    base = Path(__file__).parent

    required_dirs = [
        "data/genres/rock",
        "data/genres/jazz",
        "data/genres/classical",
        "data/genres/hiphop",
        "data/genres/pop",
        "data/genres/blues",
        "data/genres/country",
        "data/genres/disco",
        "data/genres/metal",
        "data/genres/reggae",
        "logs"
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = base / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path}")
            all_ok = False

    return all_ok


def check_files():
    """Check required Python files."""
    print("\nChecking Python files:")
    base = Path(__file__).parent

    required_files = [
        "config.py",
        "dataset_loader.py",
        "signal_utils.py",
        "setup_dataset.py",
        "examples.py",
        "requirements.txt"
    ]

    all_ok = True
    for file_name in required_files:
        file_path = base / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name}")
            all_ok = False

    return all_ok


def main():
    print("="*70)
    print("PROJECT SETUP VERIFICATION")
    print("="*70 + "\n")

    checks = [
        ("Python Version", check_python_version),
        ("Required Libraries", check_imports),
        ("Directory Structure", check_directory_structure),
        ("Python Files", check_files)
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"Error during {check_name}: {e}")
            results.append(False)

    print("\n" + "="*70)
    if all(results):
        print("✓ ALL CHECKS PASSED - Setup is complete!")
        print("\nNext steps:")
        print("  1. Download dataset: python setup_dataset.py")
        print("  2. Load dataset: python dataset_loader.py")
        print("  3. Run examples: python examples.py")
    else:
        print("✗ Some checks failed - Please review the output above")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
