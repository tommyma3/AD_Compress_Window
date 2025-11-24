"""
Quick Start Script for Attention Visualization

This script guides you through the process of visualizing attention patterns
and designing a compression strategy for your Algorithm Distillation model.
"""

import os
import sys
from pathlib import Path

def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dependencies():
    """Check if all required packages are available."""
    print("Checking dependencies...")
    missing = []
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'einops': 'Einops',
        'h5py': 'HDF5',
    }
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies are installed!")
    return True

def check_model():
    """Check if model checkpoint exists."""
    print("\nChecking for trained model...")
    
    # Look for runs directory
    runs_dir = Path('./runs')
    if not runs_dir.exists():
        print("  ✗ No 'runs' directory found")
        print("\n⚠ You need to train the model first!")
        print("  Run: python train.py")
        return False
    
    # Look for checkpoints
    ckpt_files = list(runs_dir.glob('*/ckpt-*.pt'))
    if not ckpt_files:
        print("  ✗ No checkpoint files found")
        print("\n⚠ You need to train the model first!")
        print("  Run: python train.py")
        return False
    
    latest_ckpt = sorted(ckpt_files)[-1]
    print(f"  ✓ Found checkpoint: {latest_ckpt}")
    return True

def check_data():
    """Check if dataset exists."""
    print("\nChecking for dataset...")
    
    datasets_dir = Path('./datasets')
    if not datasets_dir.exists():
        print("  ✗ No 'datasets' directory found")
        return False
    
    hdf5_files = list(datasets_dir.glob('*.hdf5'))
    if not hdf5_files:
        print("  ✗ No HDF5 dataset files found")
        return False
    
    print(f"  ✓ Found {len(hdf5_files)} dataset file(s)")
    return True

def print_workflow():
    """Print the recommended workflow."""
    print("\n📋 RECOMMENDED WORKFLOW")
    print("-" * 70)
    print("\n1. Setup and Check")
    print("   → This script (you are here!)")
    print("   → Ensures everything is ready")
    print("\n2. Visualize Attention")
    print("   → Run: python visualize_attention.py")
    print("   → Analyzes transformer attention patterns")
    print("   → Generates visualizations and statistics")
    print("\n3. Analyze Results")
    print("   → Check console output for insights")
    print("   → Review figures in ./figs/attention/")
    print("   → Identify attention pattern type")
    print("\n4. Choose Compression Strategy")
    print("   → Based on attention patterns:")
    print("     • Recency bias → RecencyCompressor")
    print("     • Uniform attention → UniformCompressor")
    print("     • Sparse attention → AttentionBasedCompressor")
    print("     • Complex patterns → AdaptiveCompressor")
    print("\n5. Test Compression")
    print("   → Run: python compression_strategies.py")
    print("   → Tests all compression methods")
    print("\n6. Integrate and Train")
    print("   → Modify model/ad.py with chosen compressor")
    print("   → Retrain model with compressed history")
    print("   → Evaluate performance vs. full history")
    print()

def print_next_steps():
    """Print immediate next steps."""
    print("\n🚀 NEXT STEPS")
    print("-" * 70)
    print("\n1. Run the attention visualization:")
    print("   → python visualize_attention.py")
    print("\n2. While it runs, read the guides:")
    print("   → ATTENTION_VISUALIZATION_GUIDE.md")
    print("   → VISUALIZATION_SUMMARY.md")
    print("\n3. After completion, check results:")
    print("   → ./figs/attention/ (visualizations)")
    print("   → Console output (statistics and insights)")
    print("\n4. Test compression strategies:")
    print("   → python compression_strategies.py")
    print()

def main():
    """Main execution."""
    print_banner("ATTENTION VISUALIZATION - QUICK START")
    
    print("This script will help you get started with visualizing")
    print("transformer attention patterns in your Algorithm Distillation model.")
    print("\nWe'll check if everything is ready and guide you through the process.")
    
    # Check dependencies
    print_banner("STEP 1: Dependencies")
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Setup incomplete!")
        print("\nFix missing dependencies and run this script again.")
        return
    
    # Check model
    print_banner("STEP 2: Model Check")
    model_ok = check_model()
    
    # Check data
    print_banner("STEP 3: Data Check")
    data_ok = check_data()
    
    # Summary
    print_banner("SETUP SUMMARY")
    
    print("Status:")
    print(f"  {'✓' if deps_ok else '✗'} Dependencies installed")
    print(f"  {'✓' if model_ok else '✗'} Trained model available")
    print(f"  {'✓' if data_ok else '✗'} Dataset available")
    
    if deps_ok and model_ok and data_ok:
        print("\n✅ Everything is ready!")
        print_workflow()
        print_next_steps()
        
        print("\n💡 TIP: The visualization will take a few minutes to complete.")
        print("   Each layer analysis processes multiple samples and generates figures.")
        
        print("\n" + "="*70)
        print("  Ready to start? Run: python visualize_attention.py")
        print("="*70 + "\n")
        
    else:
        print("\n⚠ Setup incomplete!")
        
        if not model_ok:
            print("\n→ Train your model first:")
            print("  python train.py")
        
        if not data_ok:
            print("\n→ Ensure dataset is available in ./datasets/")
        
        print("\nOnce everything is ready, run this script again to verify.")
    
    # Additional resources
    print("\n📚 DOCUMENTATION")
    print("-" * 70)
    print("  • ATTENTION_VISUALIZATION_GUIDE.md - Detailed usage guide")
    print("  • VISUALIZATION_SUMMARY.md - Overview and technical details")
    print("  • compression_strategies.py - Ready-to-use compression methods")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
