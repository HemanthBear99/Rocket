"""
RLV Phase-I Ascent Simulation - CLI

The single entry point for running simulations, generating specific plots,
and handling configuration.
"""

import argparse
import logging
import os
import sys

from rlv_sim.main import run_simulation
import generate_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RLV Phase-I Ascent Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="plots",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()
    
    # Configure verbosity
    if args.quiet:
        logger.setLevel(logging.WARNING)
    
    print(f"\n{'='*70}\nRLV PHASE-I: MASTER SIMULATION RUN\n{'='*70}\n")
    
    try:
        # 1. Run Simulation
        logger.info("Starting simulation...")
        print(">> Running Simulation Physics Engine...")
        final_state, log, reason = run_simulation(verbose=not args.quiet)
        
        # 2. Print Summary
        print("\n" + "="*60)
        print("SIMULATION SUMMARY")
        print("="*60)
        print(f"Termination reason: {reason}")
        print(f"Final time: {final_state.t:.2f} s")
        print(f"Final altitude: {final_state.altitude/1000:.2f} km")
        print(f"Final velocity: {final_state.speed:.2f} m/s")
        print("="*60 + "\n")
        
        # 3. Generate Plots
        if not args.no_plots and len(log.time) > 0:
            # Resolve output directory
            if os.path.isabs(args.output_dir):
                plot_dir = args.output_dir
            else:
                plot_dir = os.path.join(os.getcwd(), args.output_dir)
            
            logger.info(f"Generating plots in {plot_dir}")
            print(f">> Generating Plots in: {plot_dir}")
            
            generate_plots.generate_all_plots(log, final_state, plot_dir)
            
            print("\n" + "="*70)
            print("SUCCESS: Simulation and Plotting Complete.")
            print(f"Check outputs in: {plot_dir}")
            print("="*70)
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        print(f"\n[ERROR] Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
