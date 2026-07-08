import os
import sys
import unittest

# Automatically set PYTHONPATH to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_all_tests():
    print("==================================================")
    print("          STARTING ALL TEST SUITES                ")
    print("==================================================")
    
    # Discover and load all tests in the 'test' directory
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir=test_dir, pattern="test_*.py")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("==================================================")
    print("                TEST REPORT                       ")
    print("==================================================")
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print("==================================================")
    
    if not result.wasSuccessful():
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
