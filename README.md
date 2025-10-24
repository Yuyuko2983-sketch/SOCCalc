
# SOC Calculator

A specialized quantum chemistry toolkit for calculating molecular spin-orbit coupling (SOC) and phosphorescence properties, built upon the PySCF framework.

## Features

- **Comprehensive SOC Calculations**: Compute spin-orbit coupling matrix elements between electronic states
- **Phosphorescence Properties**: Predict phosphorescence lifetimes and transition rates
- **Multi-State Support**: Batch processing for multiple state pairs
- **Visualization**: Generate professional plots for result analysis
- **Robust Error Handling**: Graceful degradation with alternative algorithms

## Theoretical Foundation

Based on multireference quantum chemical methods, this framework employs the Breit-Pauli approximation for SOC calculations:

\[H_{\text{SOC}}^{\rightarrow}=\frac{\alpha^{2}}{2}\sum_{l}\sum_{I}\frac{Z_{I}}{| \mathbf{r}_{1}-\mathbf{R}_{1}|^{3}}(\mathbf{r}_{1}-\mathbf{R}_{1})\times\nabla _{i}\cdot\mathbf{s}_{1}\]

The tool accurately describes the multireference character of intramolecular charge transfer (ICT) states through the complete active space method and incorporates vibrationally assisted effects for consistent experimental predictions.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd soc-calculator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from soc_calculator import SOCCalculator

# Initialize calculator
calculator = SOCCalculator(verbose=True)

# Define molecular structure
atom_coords = [
    ['C', (-1.91001900, -1.53994700, 0.00012000)], 
    ['C', (-2.27169600, -0.15407400, 0.00008600)],
    # ... more atoms
    ['N', (5.31503000, -0.88819600, -0.00026100)]
]

# Run calculation
results = calculator.calculate_soc_and_phosphorescence(
    atom_coords=atom_coords,
    pkl_path='casscf_results.pkl',  # CASSCF results file
    basis='6-31g',
    state_i=0,           # Ground state
    state_j=1,           # First excited state
    energy_gap=2.861,    # Energy gap in eV
    generate_plots=True  # Generate visualization plots
)

# Access results
print(f"SOC norm: {results['soc_norm_cm']:.2f} cm⁻¹")
print(f"Lifetime: {results['lifetime']:.3e} s")
print(f"Transition rate: {results['transition_rate']:.3e} s⁻¹")
```

## Input Requirements

### Molecular Structure
- Cartesian coordinates in format: `[element, (x, y, z)]`
- Standard atomic symbols and angstrom units

### CASSCF Results
- Pickle file containing complete multireference state calculation information
- Must include: molecular orbital coefficients, CI vectors, state energies, active space parameters

### Basis Set
- Recommended: `6-31g` for standard calculations
- For heavy elements: use basis sets with diffuse functions

## Advanced Usage

### Multi-State Calculations
```python
state_pairs = [(0, 1), (0, 2), (1, 2)]

for state_i, state_j in state_pairs:
    results = calculator.calculate_soc_and_phosphorescence(
        atom_coords=atom_coords,
        pkl_path=pkl_path,
        state_i=state_i,
        state_j=state_j,
        generate_plots=True
    )
```

### Automatic Energy Gap
```python
# Let program calculate energy gap from CASSCF energies
results = calculator.calculate_soc_and_phosphorescence(
    atom_coords=atom_coords,
    pkl_path=pkl_path,
    state_i=0,
    state_j=1,
    energy_gap=None  # Auto-calculate from CASSCF
)
```

## Output

The calculator returns a dictionary with:
- `soc_norm_cm`: SOC strength in cm⁻¹
- `lifetime`: Phosphorescence lifetime in seconds
- `transition_rate`: Transition rate in s⁻¹
- Additional detailed parameters

### Classification Standards

**SOC Strength:**
- Strong coupling: >10 cm⁻¹
- Medium coupling: 1-10 cm⁻¹  
- Weak coupling: <1 cm⁻¹

**Phosphorescence Lifetime:**
- Ultrafast, Fast, Medium, Slow categories

### Visualization
When `generate_plots=True`, the tool generates:
- SOC matrix heatmaps
- Orbital contribution analysis
- Vector component diagrams
- Phosphorescence property summaries
- CI wavefunction analysis
- Energy correlation diagrams
- Comprehensive reports

## Example Output
```
SOC norm: 0.24 cm⁻¹
Lifetime: 1.000e-06 s
Transition rate: 1.000e+06 s⁻¹
Classification: Fast phosphorescence
```

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License.

## Citation

If you use this software in your research, please acknowledge the authors.

## Support

For questions and support, contact:
- 3775964975@qq.com
- hykuuz6@gmail.com
```

This README provides comprehensive documentation for the SOC Calculator, including theoretical background, installation instructions, usage examples, and output descriptions in a clear, professional format suitable for researchers and computational chemists.
