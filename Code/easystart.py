#!/usr/bin/env python3
"""
SOC Calculator - Easy Start Example
A simple demonstration of how to use the SOC Calculator for spin-orbit coupling calculations.
"""

import os
import numpy as np
from soc_calculator import SOCCalculator

def easy_start_example():
    """
    A complete working example of SOC calculation with sample molecular structure.
    """
    print("=" * 60)
    print("        SOC Calculator - Easy Start Example")
    print("=" * 60)
    
    # Initialize the calculator with verbose output
    print("\n[-] Initializing SOC Calculator...")
    calculator = SOCCalculator(verbose=True)
    
    # Define a sample molecular structure (formaldehyde-like molecule)
    print("\n[-] Setting up molecular structure...")
    atom_coords = [
        ['C', ( 0.00000000,  0.00000000,  0.00000000)],
        ['O', ( 0.00000000,  0.00000000,  1.20000000)],
        ['I', ( 0.00000000,  0.94000000, -0.54000000)],
        ['H', ( 0.00000000, -0.94000000, -0.54000000)]
    ]
    
    print(f"[..] Molecular structure: {len(atom_coords)} atoms")
    for atom in atom_coords:
        print(f"    {atom[0]:2s} {atom[1][0]:8.3f} {atom[1][1]:8.3f} {atom[1][2]:8.3f}")
    
    # For this example, we'll create a dummy CASSCF file if it doesn't exist
    # In real usage, you should provide your actual CASSCF results file
    pkl_path = 'sample_casscf_results.pkl'
    
    print(f"\n[-] CASSCF results file: {pkl_path}")
    print("[..] Note: In real usage, provide your actual CASSCF results file")
    
    # Example calculation parameters
    basis_set = '6-31g'
    initial_state = 0  # Ground state (S0)
    final_state = 1    # First excited state (T1)
    energy_gap = 3.5   # Example energy gap in eV
    
    print(f"\n[-] Calculation parameters:")
    print(f"    Basis set: {basis_set}")
    print(f"    States: {initial_state} -> {final_state}")
    print(f"    Energy gap: {energy_gap} eV")
    
    try:
        # Run the SOC calculation
        print("\n[-] Running SOC calculation...")
        print("[..] This may take a few minutes...")
        
        results = calculator.calculate_soc_and_phosphorescence(
            atom_coords=atom_coords,
            pkl_path=pkl_path,
            basis=basis_set,
            state_i=initial_state,
            state_j=final_state,
            energy_gap=energy_gap,
            generate_plots=True
        )
        
        # Display results
        print("\n" + "=" * 60)
        print("           CALCULATION RESULTS")
        print("=" * 60)
        
        print(f"\n[=] SOC Strength:      {results.get('soc_norm_cm', 'N/A'):.3f} cm⁻¹")
        print(f"[=] Transition Rate:  {results.get('transition_rate', 'N/A'):.3e} s⁻¹")
        print(f"[=] Lifetime:         {results.get('lifetime', 'N/A'):.3e} s")
        print(f"[=] Energy Gap:       {results.get('energy_gap_cm', 'N/A'):.1f} cm⁻¹")
        
        # Classification
        soc_strength = results.get('soc_norm_cm', 0)
        if soc_strength > 10:
            strength_class = "Strong"
        elif soc_strength > 1:
            strength_class = "Medium"
        else:
            strength_class = "Weak"
        
        print(f"[=] SOC Classification: {strength_class} coupling")
        
        # Additional details if available
        if 'soc_components' in results:
            print(f"\n[..] SOC Components:")
            comp = results['soc_components']
            print(f"     Vx: {comp[0]:.3e} cm⁻¹")
            print(f"     Vy: {comp[1]:.3e} cm⁻¹") 
            print(f"     Vz: {comp[2]:.3e} cm⁻¹")
        
        print(f"\n[=] Results saved to: my_phosphorescence_results.pkl")
        print(f"[=] Plots generated in current directory")
        
    except FileNotFoundError:
        print(f"\n[!] Error: CASSCF results file '{pkl_path}' not found.")
        print("\n[..] To run this example properly:")
        print("     1. Perform CASSCF calculation on your molecule")
        print("     2. Save results as 'sample_casscf_results.pkl'")
        print("     3. Run this script again")
        
    except Exception as e:
        print(f"\n[!] An error occurred: {e}")
        print("\n[..] Troubleshooting tips:")
        print("     - Check if all dependencies are installed")
        print("     - Verify your CASSCF results file format")
        print("     - Ensure molecular coordinates are correct")

def multi_state_example():
    """
    Example of calculating SOC for multiple state pairs.
    """
    print("\n" + "=" * 60)
    print("        Multi-State SOC Calculation Example")
    print("=" * 60)
    
    # Initialize calculator
    calculator = SOCCalculator(verbose=False)
    
    # Sample molecular structure (benzene-like ring)
    atom_coords = [
        ['C', ( 0.0000,  1.4000, 0.0000)],
        ['C', ( 1.2124,  0.7000, 0.0000)],
        ['C', ( 1.2124, -0.7000, 0.0000)],
        ['C', ( 0.0000, -1.4000, 0.0000)],
        ['C', (-1.2124, -0.7000, 0.0000)],
        ['C', (-1.2124,  0.7000, 0.0000)],
        ['H', ( 0.0000,  2.4800, 0.0000)],
        ['I', ( 2.1476,  1.2400, 0.0000)],
        ['H', ( 2.1476, -1.2400, 0.0000)],
        ['H', ( 0.0000, -2.4800, 0.0000)],
        ['H', (-2.1476, -1.2400, 0.0000)],
        ['H', (-2.1476,  1.2400, 0.0000)]
    ]
    
    print(f"[-] Molecular structure: {len(atom_coords)} atoms")
    
    # Define multiple state pairs to calculate
    state_pairs = [
        (0, 1),  # S0 -> T1
        (0, 2),  # S0 -> T2  
        (1, 2),  # T1 -> T2
    ]
    
    pkl_path = 'sample_casscf_results.pkl'
    
    print(f"\n[-] Calculating SOC for {len(state_pairs)} state pairs:")
    
    for i, (state_i, state_j) in enumerate(state_pairs):
        print(f"\n[..] [{i+1}/{len(state_pairs)}] States {state_i} -> {state_j}:")
        
        try:
            results = calculator.calculate_soc_and_phosphorescence(
                atom_coords=atom_coords,
                pkl_path=pkl_path,
                state_i=state_i,
                state_j=state_j,
                energy_gap=3.0 + i * 0.5,  # Example energy gaps
                generate_plots=False  # Turn off plots for faster batch processing
            )
            
            soc = results.get('soc_norm_cm', 0)
            rate = results.get('transition_rate', 0)
            lifetime = results.get('lifetime', 0)
            
            print(f"[=] SOC: {soc:.3f} cm⁻¹ | Rate: {rate:.3e} s⁻¹ | Lifetime: {lifetime:.3e} s")
            
        except Exception as e:
            print(f"[!] Error: {e}")

def quick_calculation(mol_coords, casscf_file, state_i=0, state_j=1, energy_gap=None):
    """
    Quick calculation function for direct use.
    
    Parameters:
    -----------
    mol_coords : list
        Molecular coordinates in format [['element', (x, y, z)], ...]
    casscf_file : str
        Path to CASSCF results pickle file
    state_i, state_j : int
        Initial and final states for SOC calculation
    energy_gap : float or None
        Energy gap in eV (if None, calculated from CASSCF)
    
    Returns:
    --------
    dict : Calculation results
    """
    print(f"[-] Starting quick SOC calculation...")
    print(f"[..] States: {state_i} -> {state_j}")
    
    calculator = SOCCalculator(verbose=True)
    
    results = calculator.calculate_soc_and_phosphorescence(
        atom_coords=mol_coords,
        pkl_path=casscf_file,
        state_i=state_i,
        state_j=state_j,
        energy_gap=energy_gap,
        generate_plots=True
    )
    
    return results

if __name__ == "__main__":
    print("SOC Calculator Easy Start")
    print("This example demonstrates basic usage of the SOC Calculator.")
    print("For real calculations, provide your actual CASSCF results file.\n")
    
    # Run the main example
    easy_start_example()
    
    # Uncomment the line below to run the multi-state example
    # multi_state_example()
    
    print("\n" + "=" * 60)
    print("[=] Example completed!")
    print("[=] For more information, see the full documentation.")
    print("=" * 60)
	