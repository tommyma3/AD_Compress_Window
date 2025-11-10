"""
Quick start script for attention analysis.
Runs both evaluation and analysis in sequence.
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*80)
    print(f"{description}")
    print("="*80)
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with return code {result.returncode}")
        return False
    return True

def main():
    print("\n" + "="*80)
    print("ATTENTION ANALYSIS PIPELINE")
    print("="*80)
    
    # Check if checkpoint exists
    ckpt_dir = Path('./runs/AD-darkroom-seed0')
    if not ckpt_dir.exists():
        print(f"\nError: Checkpoint directory not found: {ckpt_dir}")
        print("Please train a model first or update the checkpoint path.")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path('./figs/attention_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Capture attention during evaluation
    if not run_command(
        "python evaluate_with_attention.py",
        "Step 1: Capturing attention patterns during evaluation"
    ):
        print("\nFailed to capture attention data. Please check the error above.")
        sys.exit(1)
    
    # Check if data was generated
    data_file = output_dir / 'attention_data.npz'
    if not data_file.exists():
        print(f"\nError: Attention data file not created: {data_file}")
        sys.exit(1)
    
    print(f"\n✓ Attention data saved to: {data_file}")
    
    # Step 2: Analyze and visualize
    if not run_command(
        "python analyze_attention.py",
        "Step 2: Analyzing attention patterns and generating visualizations"
    ):
        print("\nFailed to analyze attention data. Please check the error above.")
        sys.exit(1)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - attention_heatmap_step*.png  (attention at different timesteps)")
    print(f"  - attention_analysis_comprehensive.png  (main analysis)")
    print(f"  - layer_head_analysis.png  (per-layer/head patterns)")
    print(f"  - compression_recommendations.txt  (strategy recommendations)")
    print(f"  - attention_data.npz  (raw data for further analysis)")
    
    print("\nNext steps:")
    print("  1. Review the visualizations in the output directory")
    print("  2. Read compression_recommendations.txt for strategies")
    print("  3. Implement compression in model/ad.py based on insights")
    print("  4. Test inference speedup vs performance tradeoff")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
