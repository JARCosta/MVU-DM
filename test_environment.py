#!/usr/bin/env python3
"""
Test script to verify the Python environment and required packages.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_package_imports():
    """Test importing all required packages."""
    print("Testing package imports...")
    
    packages = [
        ('matlabengine', '24.2.2'),
        ('sklearn', '1.6.1'),
        ('scipy', '1.15.2'),
        ('matplotlib', '3.10.3'),
        ('shapely', '2.0.2')
    ]
    
    results = []
    
    for package_name, expected_version in packages:
        try:
            if package_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
                print(f"✓ scikit-learn: {version}")
            elif package_name == 'matlabengine':
                import matlab.engine
                # Note: matlabengine version might not be directly accessible
                print(f"✓ matlabengine: imported successfully")
            elif package_name == 'scipy':
                import scipy
                version = scipy.__version__
                print(f"✓ scipy: {version}")
            elif package_name == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
                print(f"✓ matplotlib: {version}")
            elif package_name == 'shapely':
                import shapely
                version = shapely.__version__
                print(f"✓ shapely: {version}")
            
            results.append((package_name, True, version if 'version' in locals() else 'unknown'))
            
        except ImportError as e:
            print(f"✗ {package_name}: Failed to import - {e}")
            results.append((package_name, False, str(e)))
        except Exception as e:
            print(f"? {package_name}: Imported with warning - {e}")
            results.append((package_name, True, f"warning: {e}"))
    
    return results

def test_basic_functionality():
    """Test basic functionality of key packages."""
    print("\nTesting basic functionality...")
    
    try:
        # Test numpy/scipy
        import numpy as np
        import scipy.stats as stats
        
        # Create test data
        data = np.random.randn(100)
        mean = np.mean(data)
        std = np.std(data)
        print(f"✓ NumPy/SciPy: Generated random data (mean={mean:.3f}, std={std:.3f})")
        
        # Test scikit-learn
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"✓ Scikit-learn: Random Forest accuracy = {score:.3f}")
        
        # Test matplotlib (without displaying)
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.close(fig)
        print("✓ Matplotlib: Plot created successfully")
        
        # Test shapely
        from shapely.geometry import Point, Polygon
        
        point = Point(0.5, 0.5)
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        contains = polygon.contains(point)
        print(f"✓ Shapely: Point in polygon test = {contains}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("PYTHON ENVIRONMENT TEST")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("=" * 60)
    
    # Test imports
    import_results = test_package_imports()
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_packages = len(import_results)
    
    print(f"Package imports: {successful_imports}/{total_packages} successful")
    print(f"Functionality tests: {'PASSED' if functionality_ok else 'FAILED'}")
    
    if successful_imports == total_packages and functionality_ok:
        print("\n🎉 Environment is ready for use!")
        return 0
    else:
        print("\n⚠️  Some issues detected. Check the output above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)