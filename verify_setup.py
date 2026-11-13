"""
Setup Verification
"""

import sys
import os

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    print(f"✓ Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  WARNING: Python 3.8 or higher is recommended")
        return False
    print("  ✓ Version OK")
    return True


def check_packages():
    """Check if required packages are installed"""
    print("\n=== Checking Required Packages ===")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'plotly',
        'flask',
        'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required packages are installed!")
    return True


def check_directory_structure():
    """Check if all required directories exist"""
    print("\n=== Checking Directory Structure ===")
    
    required_dirs = [
        'dataset',
        'src',
        'app',
        'results',
        'app/templates',
        'app/static'
    ]
    
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✓ All directories are present!")
    else:
        print("\n⚠️  Some directories are missing")
    
    return all_exist


def check_dataset():
    """Check if dataset is present"""
    print("\n=== Checking Dataset ===")
    
    dataset_paths = [
        'dataset/road_accidents_india_2017_2022.csv',
        'dataset/Road.csv',
        'dataset/processed_data.csv'
    ]
    
    dataset_found = False
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"  ✓ Found: {path}")
            dataset_found = True
            break
    
    if not dataset_found:
        print("  ⚠️  Dataset not found!")
        print("  Please place your dataset in the dataset/ folder")
        print("  Expected name: road_accidents_india_2017_2022.csv")
        return False
    
    return True


def check_files():
    """Check if all required files exist"""
    print("\n=== Checking Project Files ===")
    
    required_files = [
        'src/data_preprocessing.py',
        'src/model_training.py',
        'src/model_evaluation.py',
        'src/utils.py',
        'src/predict.py',
        'app/app.py'
    ]
    
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NOT FOUND")
            all_exist = False
    
    if all_exist:
        print("\n✓ All required files are present!")
    else:
        print("\n⚠️  Some files are missing")
    
    return all_exist


def check_disk_space():
    """Check available disk space"""
    print("\n=== Checking Disk Space ===")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        
        free_gb = free // (2**30)
        print(f"  Available: {free_gb} GB")
        
        if free_gb < 2:
            print("  ⚠️  WARNING: Less than 2 GB free space")
            print("  Consider freeing up some disk space")
            return False
        else:
            print("  ✓ Sufficient disk space")
            return True
    except:
        print("  ⚠️  Could not check disk space")
        return True


def main():
    """Run all checks"""
    print("="*60)
    print("ROAD ACCIDENT SEVERITY PREDICTION - SETUP VERIFICATION")
    print("="*60)
    
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_packages(),
        "Directory Structure": check_directory_structure(),
        "Project Files": check_files(),
        "Dataset": check_dataset(),
        "Disk Space": check_disk_space()
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*60)
    
    if all_passed:
        print("✓ SETUP COMPLETE - YOU'RE READY TO START!")
        print("\nNext steps:")
        print("1. Ensure dataset is in dataset/ folder")
        print("2. Run: python src/data_preprocessing.py")
        print("3. Run: python src/model_training.py")
        print("4. Run: python app/app.py")
    else:
        print("⚠️  SETUP INCOMPLETE")
        print("\nPlease fix the issues")
    print("="*60)
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
