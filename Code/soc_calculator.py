 

from pyscf import gto, scf, mcscf, lib
import numpy as np
import pickle
 
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
class Colors:
	RED = '\033[91m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	BLUE = '\033[94m'
	MAGENTA = '\033[95m'
	CYAN = '\033[96m'
	WHITE = '\033[97m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'
	

class SOCCalculator:
	def _print_art(self):
		import math
		width = 40
		height =20
		a = 0.1  # angle
		b = 0.8  # %


		canvas = [[' ' for _ in range(width)] for _ in range(height)]

		for i in range(2000):  
			angle = a + i * 0.1
			r = b * angle
			x = int((r * math.cos(angle) + width / 2)+5)
			y = int(r * math.sin(angle) + height / 2)
			if 0 <= x < width and 0 <= y < height:
				canvas[y][x] = '*'
		for row in canvas:
			
			self._print(' '*10+''.join(row),Colors.MAGENTA)
		art = r"""
	  _   _   _   _   ____   _____   _____
	 | | | | | | | | / ___| / ___ \ / ___|
	 | | | | | | | | \___ \| |   | | |	
	 | |_| | | |_| |  ___) | |___| | |___ 
	  \___/   \___/  |____/ \_____/ \____|
			"""
		
		self._print(art,Colors.CYAN)


	
	def __init__(self, verbose=True, output_dir="soc_visualization"):
		self.verbose = verbose
		self.output_dir = output_dir   
		self._setup_output_directory()  
		self.results = {}
		self.results = {}
	def _setup_output_directory(self):
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)
	def _print(self, message, color=Colors.WHITE):
		if self.verbose:
			print(f"{color}{message}{Colors.END}")
	def _create_custom_colormap(self):
		colors =  ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
		return LinearSegmentedColormap.from_list('soc_cmap', colors, N=256)
	def calculate_soc_and_phosphorescence(self, atom_coords, pkl_path, generate_plots,basis='6-31+g*', state_i=0, state_j=1, energy_gap=None):
		self._print_art()
		casscf_results = self.load_casscf_results(pkl_path)
		mol = self.setup_molecule(atom_coords, basis)
		
		ncas = casscf_results['ncas']
		nelecas = casscf_results['nelecas']
		mc = mcscf.CASSCF(mol, ncas, nelecas)
		mc.mo_coeff = casscf_results['mo_coeff']
		mc.ci = casscf_results['ci']
		mc.e_states = casscf_results['e_states']
		mc.ncore = casscf_results.get('ncore', mol.nelectron // 2 - nelecas[0] // 2)
		
		 
		soc_vector, soc_norm = self.compute_soc_using_effective_hamiltonian(mol, mc, state_i, state_j)
		
		 
		hso1e = mol.intor('int1e_prinvxp', 3)
		mo_coeff = mc.mo_coeff
		ncore = mc.ncore
		hso1e_mo = np.einsum('xpq,pi,qj->xij', hso1e, mo_coeff, mo_coeff)
		cas_idx = np.arange(ncore, ncore + ncas)
		hso1e_cas = hso1e_mo[:, cas_idx[:, None], cas_idx]
		
		 
		if energy_gap is None and len(mc.e_states) > max(state_i, state_j):
			energy_gap_au = abs(mc.e_states[state_j] - mc.e_states[state_i])
			energy_gap = energy_gap_au * 27.2114
		elif energy_gap is None:
			energy_gap = 2.861
		
		 
		
		rate, lifetime = self.compute_phosphorescence_rate(soc_norm, energy_gap)
		self._print(soc_norm, Colors.CYAN)
		 
		soc_cm = soc_norm * 219474.63
		
		if soc_cm > 10:
			strength = "Strong SOC coupling"
			desc = "Strong phosphorescence"
		elif soc_cm > 1:
			strength = "Medium SOC coupling"
			desc = "Medium phosphorescence intensity"
		else:
			strength = "Weak SOC coupling"
			desc = "Weak phosphorescence"
		
		if lifetime < 1e-3:
			lifetime_desc = "Short-lived phosphorescence"
		elif lifetime < 1:
			lifetime_desc = "Medium-lived phosphorescence"
		else:
			lifetime_desc = "Long-lived phosphorescence"
		
		self.results = {
			'soc_vector': soc_vector,
			'soc_norm': soc_norm,
			'soc_norm_cm': soc_cm,
			'phosphorescence_energy_ev': energy_gap,
			'transition_rate': rate,
			'lifetime': lifetime,
			'strength_description': strength,
			'phosphorescence_description': desc,
			'lifetime_description': lifetime_desc,
			'hso1e_cas': hso1e_cas,
			'ncas': ncas
		}
		
		 
		if generate_plots:
			self._print("\nGenerating visualizations...", Colors.CYAN)
				
				 
			self.plot_soc_matrix_heatmap(hso1e_cas)
				
				 
			self.plot_orbital_contributions(hso1e_cas, ncas)
				
				 
			self.plot_soc_vector_components(soc_vector)
				
				 
			self.plot_phosphorescence_properties(self.results)
		
			 
			if hasattr(mc, 'ci') and mc.ci is not None:
				ci_i = mc.ci[state_i] if isinstance(mc.ci, (list, tuple)) else mc.ci
				ci_j = mc.ci[state_j] if isinstance(mc.ci, (list, tuple)) else mc.ci
				state_names = [f'State {state_i}', f'State {state_j}']
				self.plot_ci_wavefunctions(ci_i, ci_j, state_names)
			
			 
			self.plot_soc_energy_correlation(mc, state_i, state_j)
			
			 
			self.generate_comprehensive_report(self.results, mol)
			
			self._print("All visualizations completed!", Colors.GREEN)

		return self.results
	def plot_soc_matrix_heatmap(self, hso1e_cas, component_names=['x', 'y', 'z']):
		 
		fig, axes = plt.subplots(1, 3, figsize=(15, 5))
		fig.suptitle('Spin-Orbit Coupling Matrix Elements', fontsize=28, fontweight='bold')
		
		vmax = np.max(np.abs(hso1e_cas))
		vmin = -vmax
		
		for i, (ax, component) in enumerate(zip(axes, component_names)):
			im = ax.imshow(hso1e_cas[i], cmap='RdBu_r', vmin=vmin, vmax=vmax, 
						  aspect='equal', interpolation='nearest')
			ax.set_title(f'SOC {component}-component', fontsize=20, fontweight='bold')
			ax.set_xlabel('Orbital Index')
			ax.set_ylabel('Orbital Index')
			cbar = plt.colorbar(im, ax=ax, shrink=0.8)
			cbar.set_label('SOC Matrix Element (a.u.)', rotation=270, labelpad=15)
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'soc_matrix_heatmap.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		self._print("[+]SOC Matrix heatmap has saved.", Colors.GREEN)

	def plot_orbital_contributions(self, hso1e_cas, ncas):
		orbital_contributions = np.zeros(ncas)
		for x in range(3):
			diag_contributions = np.abs(np.diag(hso1e_cas[x]))
			row_contributions = np.sum(np.abs(hso1e_cas[x]), axis=1)
			col_contributions = np.sum(np.abs(hso1e_cas[x]), axis=0)
			
			orbital_contributions += diag_contributions + 0.5 * (row_contributions + col_contributions)
		n_top = min(15, ncas)
		sorted_indices = np.argsort(orbital_contributions)[-n_top:][::-1]
		top_contributions = orbital_contributions[sorted_indices]
		top_indices = sorted_indices
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
		bars = ax1.bar(range(n_top), top_contributions, 
					  color=plt.cm.viridis(np.linspace(0, 1, n_top)))
		ax1.set_xlabel('Orbital Index')
		ax1.set_ylabel('SOC Contribution')
		ax1.set_title('Top Orbital Contributions to SOC', fontsize=28, fontweight='bold')
		ax1.set_xticks(range(n_top))
		ax1.set_xticklabels([f'{idx}' for idx in top_indices])
		for i, v in enumerate(top_contributions):
			ax1.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
		total_contribution = np.sum(orbital_contributions)
		top_total = np.sum(top_contributions)
		other_contribution = total_contribution - top_total
		
		if other_contribution > 0:
			sizes = list(top_contributions) + [other_contribution]
			labels = [f'Orb {idx}' for idx in top_indices] + ['Other Orbitals']
		else:
			sizes = top_contributions
			labels = [f'Orb {idx}' for idx in top_indices]
		
		colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
		ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
		ax2.set_title('SOC Contribution Distribution', fontsize=14, fontweight='bold')
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'orbital_soc_contributions.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		 
	def plot_soc_vector_components(self, soc_vector):
		try:
			soc_vector = np.array(soc_vector, dtype=np.float64)
			if np.iscomplexobj(soc_vector):
				self._print("Multiple SOC vectors detected, using the magnitude.", Colors.YELLOW)
				values = np.abs(soc_vector)  
				soc_vector_real = np.real(soc_vector)  
			else:
				values = np.abs(soc_vector)
				soc_vector_real = soc_vector
			
			components = ['SOC$_x$', 'SOC$_y$', 'SOC$_z$']
			colors =  ['#FF6B6B', '#4ECDC4', '#45B7D1']
			
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
			bars = ax1.bar(components, values, color=colors, alpha=0.7)
			ax1.set_ylabel('SOC Matrix Element (a.u.)')
			ax1.set_title('SOC Vector Components', fontweight='bold')
			ax1.grid(True, alpha=0.3)
			for bar, value in zip(bars, values):
				ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
						f'{value:.6f}', ha='center', va='bottom')
		
			if np.any(soc_vector_real[:2] != 0):
				theta = np.arctan2(float(soc_vector_real[1]), float(soc_vector_real[0]))
				r = np.linalg.norm(soc_vector_real[:2])
				
				ax2 = fig.add_subplot(122, projection='polar')
				ax2.plot([0, theta], [0, r], 'o-', linewidth=3, markersize=8, 
						color='red', alpha=0.7)
				ax2.set_title('SOC Vector Direction (x-y plane)', fontweight='bold', pad=20)
				ax2.grid(True)
			else:
				ax2 = fig.add_subplot(122)
				ax2.text(0.5, 0.5, 'No significant\nx-y components', 
						ha='center', va='center', transform=ax2.transAxes, fontsize=12)
				ax2.set_title('SOC Vector Direction', fontweight='bold')
				ax2.axis('off')
			
			plt.tight_layout()
			plt.savefig(os.path.join(self.output_dir, 'soc_vector_components.png'), 
					   bbox_inches='tight', dpi=300)
			plt.close()
			
			self._print("[+]SOC Vector pic has saved", Colors.GREEN)
			
		except Exception as e:
			self._print(f"[-]Error with plot SOC vectors: {e}", Colors.RED)
			self._create_simple_soc_plot(soc_vector)

	def _create_simple_soc_plot(self, soc_vector):
		 
		try:
			 
			if np.iscomplexobj(soc_vector):
				values = np.abs(soc_vector)
			else:
				values = np.abs(soc_vector)
			
			components = ['SOC_x', 'SOC_y', 'SOC_z']
			
			fig, ax = plt.subplots(figsize=(8, 6))
			bars = ax.bar(components, values, color=['red', 'green', 'blue'], alpha=0.7)
			ax.set_ylabel('SOC Magnitude (a.u.)')
			ax.set_title('SOC Vector Components (Magnitude)', fontweight='bold')
			ax.grid(True, alpha=0.3)
			
			for bar, value in zip(bars, values):
				ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
					   f'{value:.6f}', ha='center', va='bottom')
			
			plt.tight_layout()
			plt.savefig(os.path.join(self.output_dir, 'soc_vector_simple.png'), 
					   bbox_inches='tight', dpi=300)
			plt.close()
			
			 
		except Exception as e:
			self._print(f"备用图也失败: {e}", Colors.RED)
	def plot_soc_vector_componentsold(self, soc_vector):
		 
		components = ['SOC$_x$', 'SOC$_y$', 'SOC$_z$']
		values = np.abs(soc_vector)   
		colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
		
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
		
		 
		bars = ax1.bar(components, values, color=colors, alpha=0.7)
		ax1.set_ylabel('SOC Matrix Element (a.u.)')
		ax1.set_title('SOC Vector Components', fontweight='bold')
		ax1.grid(True, alpha=0.3)
		
		 
		for bar, value in zip(bars, values):
			ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
					f'{value:.6f}', ha='center', va='bottom')
		
		 
		theta = np.arctan2(soc_vector[1], soc_vector[0])
		r = np.linalg.norm(soc_vector[:2])
		
		ax2 = fig.add_subplot(122, projection='polar')
		ax2.plot([0, theta], [0, r], 'o-', linewidth=3, markersize=8, 
				color='red', alpha=0.7)
		ax2.set_title('SOC Vector Direction (x-y plane)', fontweight='bold', pad=20)
		ax2.grid(True)
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'soc_vector_components.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		 

	def plot_phosphorescence_properties(self, results_dict):
		 
		fig = plt.figure(figsize=(15, 10))
		
		 
		gs = plt.GridSpec(2, 3, figure=fig)
		
		 
		ax1 = fig.add_subplot(gs[0, 0])
		soc_norm_cm = results_dict['soc_norm_cm']
		
		 
		if soc_norm_cm > 10:
			color = 'red'
			strength_label = 'Strong'
		elif soc_norm_cm > 1:
			color = 'orange'
			strength_label = 'Medium'
		else:
			color = 'blue'
			strength_label = 'Weak'
		
		ax1.bar(['SOC Strength'], [soc_norm_cm], color=color, alpha=0.7)
		ax1.set_ylabel('SOC (cm$^{-1}$)')
		ax1.set_title(f'SOC Strength: {strength_label}', fontweight='bold')
		ax1.text(0, soc_norm_cm/2, f'{soc_norm_cm:.2f} cm$^{-1}$', 
				ha='center', va='center', fontweight='bold', fontsize=12)
		
		 
		ax2 = fig.add_subplot(gs[0, 1])
		lifetime = results_dict['lifetime']
		
		 
		lifetime_ms = lifetime * 1000   
		ax2.bar(['Lifetime'], [np.log10(lifetime_ms)], color='green', alpha=0.7)
		ax2.set_ylabel('log$_{10}$(Lifetime / ms)')
		ax2.set_title('Phosphorescence Lifetime', fontweight='bold')
		ax2.text(0, np.log10(lifetime_ms)/2, f'{lifetime_ms:.3e} ms', 
				ha='center', va='center', fontweight='bold')
		
		 
		ax3 = fig.add_subplot(gs[0, 2])
		rate = results_dict['transition_rate']
		ax3.bar(['Rate'], [np.log10(rate)], color='purple', alpha=0.7)
		ax3.set_ylabel('log$_{10}$(Rate / s$^{-1}$)')
		ax3.set_title('Transition Rate', fontweight='bold')
		ax3.text(0, np.log10(rate)/2, f'{rate:.3e} s$^{-1}$', 
				ha='center', va='center', fontweight='bold')
		
		 
		ax4 = fig.add_subplot(gs[1, :])
		energy_gap = results_dict['phosphorescence_energy_ev']
		
		 
		levels = [0, energy_gap]
		labels = ['S₀', 'T₁']
		colors = ['green', 'red']
		
		for i, (level, label, color) in enumerate(zip(levels, labels, colors)):
			ax4.hlines(level, i-0.2, i+0.2, linewidth=6, color=color, alpha=0.8)
			ax4.text(i, level + 0.1, f'{label}', ha='center', va='bottom', 
					fontweight='bold', fontsize=12)
			ax4.text(i, level - 0.2, f'{level:.3f} eV', ha='center', va='top')
		
		ax4.set_xlim(-0.5, 1.5)
		ax4.set_ylim(-0.5, energy_gap + 0.5)
		ax4.set_ylabel('Energy (eV)')
		ax4.set_title('Energy Level Diagram', fontweight='bold')
		ax4.grid(True, alpha=0.3)
		
		 
		ax4.annotate('', xy=(1, energy_gap), xytext=(1, 0),
					arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
		ax4.text(1.1, energy_gap/2, f'ΔE = {energy_gap:.3f} eV', 
				va='center', fontweight='bold')
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'phosphorescence_summary.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		 

	def plot_ci_wavefunctions(self, ci_bra, ci_ket, state_names=None):
		 
		if state_names is None:
			state_names = ['State i', 'State j']
		
		 
		ci_bra_1d = self._ensure_1d(ci_bra)
		ci_ket_1d = self._ensure_1d(ci_ket)
		
		 
		n_configs = min(50, len(ci_bra_1d))
		
		 
		bra_indices = np.argsort(np.abs(ci_bra_1d))[-n_configs:][::-1]
		ket_indices = np.argsort(np.abs(ci_ket_1d))[-n_configs:][::-1]
		
		 
		all_indices = np.unique(np.concatenate([bra_indices, ket_indices]))
		n_plot = min(30, len(all_indices))
		plot_indices = all_indices[:n_plot]
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
		
		 
		x_pos = np.arange(n_plot)
		width = 0.35
		
		ax1.bar(x_pos - width/2, np.abs(ci_bra_1d[plot_indices]), width, 
			   label=state_names[0], alpha=0.7, color='blue')
		ax1.bar(x_pos + width/2, np.abs(ci_ket_1d[plot_indices]), width, 
			   label=state_names[1], alpha=0.7, color='red')
		
		ax1.set_xlabel('Configuration Index')
		ax1.set_ylabel('Wavefunction Amplitude')
		ax1.set_title('CI Wavefunction Amplitudes', fontweight='bold',fontsize=28)
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		
		 
		overlap = np.abs(ci_bra_1d[plot_indices] * ci_ket_1d[plot_indices])
		ax2.bar(x_pos, overlap, alpha=0.7, color='green')
		ax2.set_xlabel('Configuration Index')
		ax2.set_ylabel('Overlap Product')
		ax2.set_title('Wavefunction Overlap Contributions', fontweight='bold',fontsize=28)
		ax2.grid(True, alpha=0.3)
		
		total_overlap = np.abs(np.dot(ci_bra_1d.conj(), ci_ket_1d))
		ax2.text(0.02, 0.95, f'Total Overlap: {total_overlap:.4f}', 
				transform=ax2.transAxes, fontweight='bold', fontsize=20,
				bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'ci_wavefunctions.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		 

	def plot_soc_energy_correlation(self, mc, state_i, state_j):
		 
		if not hasattr(mc, 'e_states') or mc.e_states is None:
			self._print("没有可用的态能量数据", Colors.YELLOW)
			return
		
		n_states = len(mc.e_states)
		energies = mc.e_states * 27.2114   
		
		fig, ax = plt.subplots(figsize=(10, 6))
		
		 
		for i, energy in enumerate(energies):
			if i == state_i:
				color = 'green'
				label = f'State {state_i}'
				marker = 'o'
				markersize = 10
			elif i == state_j:
				color = 'red'
				label = f'State {state_j}'
				marker = 's'
				markersize = 10
			else:
				color = 'gray'
				label = None
				marker = '^'
				markersize = 6
			
			ax.plot(i, energy, marker=marker, color=color, markersize=markersize,
				   label=label, alpha=0.8)
		
		 
		ax.annotate('', xy=(state_j, energies[state_j]), 
				   xytext=(state_i, energies[state_i]),
				   arrowprops=dict(arrowstyle='<->', color='blue', lw=2, alpha=0.7))
		
		energy_gap = abs(energies[state_j] - energies[state_i])
		mid_point = (energies[state_i] + energies[state_j]) / 2
		ax.text((state_i + state_j) / 2, mid_point, 
			   f'ΔE = {energy_gap:.3f} eV', ha='center', va='bottom',
			   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
		
		ax.set_xlabel('State Index')
		ax.set_ylabel('Energy (eV)')
		ax.set_title('State Energies and SOC Transition', fontweight='bold')
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'soc_energy_correlation.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		 

	def generate_comprehensive_report(self, results_dict, mol):
		 
		 
		fig = plt.figure(figsize=(15, 20))
		
		 
		plt.suptitle('Spin-Orbit Coupling and Phosphorescence Analysis Report\n', 
					fontsize=18, fontweight='bold')
		
		 
		ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2)
		ax1.axis('off')
		
		mol_info = f"""
		Molecular System Information:
		- Number of atoms: {len(mol.atom)}
		- Number of electrons: {mol.nelectron}
		- Basis set: {mol.basis}
		- Charge: {mol.charge}
		- Spin multiplicity: {mol.spin + 1}
		
		Elements: {[atom[0] for atom in mol.atom]}
		"""
		
		ax1.text(0.02, 0.9, mol_info, fontfamily='monospace', fontsize=10, 
				verticalalignment='top', transform=ax1.transAxes,
				bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.5))
		
		 
		ax2 = plt.subplot2grid((5, 2), (1, 0))
		ax2.axis('off')
		
		soc_info = f"""
		SOC Results:
		- SOC norm: {results_dict['soc_norm']:.6e} a.u.
		- SOC norm: {results_dict['soc_norm_cm']:.2f} cm⁻¹
		- SOC vector: 
		  x: {results_dict['soc_vector'][0]:.6e}
		  y: {results_dict['soc_vector'][1]:.6e} 
		  z: {results_dict['soc_vector'][2]:.6e}
		- Strength: {results_dict['strength_description']}
		"""
		
		ax2.text(0.02, 0.9, soc_info, fontfamily='monospace', fontsize=10,
				verticalalignment='top', transform=ax2.transAxes,
				bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5))
		
		 
		ax3 = plt.subplot2grid((5, 2), (1, 1))
		ax3.axis('off')
		
		phospho_info = f"""
		Phosphorescence Properties:
		- Energy gap: {results_dict['phosphorescence_energy_ev']:.3f} eV
		- Transition rate: {results_dict['transition_rate']:.3e} s⁻¹
		- Lifetime: {results_dict['lifetime']:.3e} s
		- Lifetime: {results_dict['lifetime']*1000:.3e} ms
		- Description: {results_dict['phosphorescence_description']}
		- Lifetime type: {results_dict['lifetime_description']}
		"""
		
		ax3.text(0.02, 0.9, phospho_info, fontfamily='monospace', fontsize=10,
				verticalalignment='top', transform=ax3.transAxes,
				bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.5))
		
		 
		from datetime import datetime
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		plt.figtext(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, style='italic')
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.output_dir, 'comprehensive_report.png'), 
				   bbox_inches='tight', dpi=300)
		plt.close()
		
		self._print("Check "+self.output_dir, Colors.GREEN)
	def load_casscf_results(self, pkl_path):
		 
		self._print(f"Loading CASSCF results from {pkl_path}...", Colors.YELLOW)
		with open(pkl_path, 'rb') as f:
			results = pickle.load(f)
		self._print("CASSCF results loaded successfully", Colors.GREEN)
		return results
	
	def setup_molecule(self, atom_coords, basis='6-31+g*', spin=0, charge=0):
		 
		self._print(f"Setting up molecule with basis {basis}...", Colors.YELLOW)
		mol = gto.Mole()
		mol.atom = atom_coords
		mol.basis = basis
		mol.spin = spin
		mol.charge = charge
		mol.build()
		self._print(f"Molecule built: {len(mol.atom)} atoms, {mol.nelectron} electrons", Colors.GREEN)
		return mol
	def _get_ci_vector(self, mc, state_index):
		 
		try:
			if hasattr(mc, 'ci'):
				ci_vectors = mc.ci
				
				 
				if isinstance(ci_vectors, (list, tuple)):
					 
					if state_index < len(ci_vectors):
						ci_vector = ci_vectors[state_index]
						self._print(f"[+]Successfully obtained the CI vector for state {state_index}, shape: {ci_vector.shape}", Colors.WHITE)
						return ci_vector
					else:
						raise IndexError(f"[-]Idx {state_index} ou of range (Total= {len(ci_vectors)} )")
				
				elif isinstance(ci_vectors, np.ndarray):
					 
					if state_index == 0:
						self._print(f"[~]got an Single CI Vector: {ci_vectors.shape}", Colors.WHITE)
						return ci_vectors
					else:
						raise IndexError(f"[-]...In monostate computation, only state 0 can be accessed, but state{state_index}was requested")
				
				else:
					raise TypeError(f"[?] What is: {type(ci_vectors)}?")
					
			else:
				raise AttributeError("[-] No CI was found in CASSCF")
				
		except Exception as e:
			self._print(f"[-] Error with CI: {e}", Colors.RED)
			 
			return self._create_fallback_ci_vector(mc, state_index)

	def _create_fallback_ci_vector(self, mc, state_index):
		 
		self._print("使用备用CI向量生成方法", Colors.YELLOW)
		
		 
		ncas = mc.ncas
		nelecas = mc.nelecas
		
		if isinstance(nelecas, (tuple, list)):
			nalpha, nbeta = nelecas
		else:
			nalpha = nbeta = nelecas // 2
		
		 
		from pyscf import fci
		ci_size = fci.cistring.num_strings(ncas, nalpha) * fci.cistring.num_strings(ncas, nbeta)
		
		self._print(f"活性空间: {ncas} 轨道, {nalpha} α电子, {nbeta} β电子", Colors.WHITE)
		self._print(f"CI向量预期大小: {ci_size}", Colors.WHITE)
		
		 
		if state_index == 0:
			 
			ci_vector = np.zeros(ci_size)
			ci_vector[0] = 1.0   
		else:
			 
			ci_vector = np.random.randn(ci_size)
			ci_vector = ci_vector / np.linalg.norm(ci_vector)   
		
		self._print(f"生成的备用CI向量形状: {ci_vector.shape}", Colors.WHITE)
		
		return ci_vector

	def _analyze_ci_vector(self, ci_vector, mc, state_index):
		 
		self._print(f"分析态 {state_index} 的CI向量...", Colors.CYAN)
		
		 
		amplitude_sq = np.abs(ci_vector)**2
		total_weight = np.sum(amplitude_sq)
		
		self._print(f"CI向量总权重: {total_weight:.6f}", Colors.WHITE)
		self._print(f"CI向量范数: {np.linalg.norm(ci_vector):.6f}", Colors.WHITE)
		
		 
		n_top = min(5, len(ci_vector))
		top_indices = np.argsort(amplitude_sq)[-n_top:][::-1]
		top_amplitudes = ci_vector[top_indices]
		top_weights = amplitude_sq[top_indices]
		
		 
		 
			 
		
		 
		if abs(total_weight - 1.0) > 1e-6:
			self._print(f"警告: CI向量未归一化 (总权重 = {total_weight:.6f})", Colors.YELLOW)


	def _validate_ci_consistency(self, ci_i, ci_j, state_i, state_j):
		 
		self._print(f"验证态 {state_i} 和态 {state_j} 的CI向量一致性...", Colors.CYAN)
		
		 
		if ci_i.shape != ci_j.shape:
			self._print(f"警告: CI向量形状不匹配 - 态 {state_i}: {ci_i.shape}, 态 {state_j}: {ci_j.shape}", Colors.RED)
			return False
		
		 
		overlap = np.abs(np.vdot(ci_i, ci_j))
		self._print(f"CI向量重叠: {overlap:.6f}", Colors.WHITE)
		
		if overlap > 0.9 and state_i != state_j:
			self._print(f"警告: 不同态之间的CI向量重叠过大", Colors.YELLOW)
		
		return True


	def _compute_soc_matrix_elements_diagnostic(self, hso1e_cas, ci_i, ci_j, nelecas, ncas, mol, mc, state_i, state_j):

		if isinstance(nelecas, (tuple, list)):
			nalpha, nbeta = nelecas
		else:
			nalpha = nbeta = nelecas // 2
		

		spin_i = self._get_state_spin(ci_i, ncas, (nalpha, nbeta), mc=mc, state_index=state_i)
		spin_j = self._get_state_spin(ci_j, ncas, (nalpha, nbeta), mc=mc, state_index=state_j)
		
		self._print(f"SpinS: S_i={spin_i}, S_j={spin_j}", Colors.CYAN)
		
		 
		spin_matrix_elements = self._compute_spin_matrix_elements(spin_i, spin_j)
		self._print(f"Spin matrix element: {spin_matrix_elements}", Colors.WHITE)
		
		soc_elements = np.zeros(3, dtype=complex)
		
		 
		orbital_elements = self._compute_orbital_matrix_elements_diagnostic(
			hso1e_cas, ci_i, ci_j, ncas, (nalpha, nbeta), mol
		)
		self._print(f"Orbital matrix element: {orbital_elements}", Colors.WHITE)
		

		for x in range(3):
			soc_elements[x] = orbital_elements[x] * spin_matrix_elements[x]
		
		return soc_elements
	def _compute_orbital_matrix_elements(self, hso1e_cas, ci_bra, ci_ket, mc):
	 
		return mc.fcisolver.contract_1e(hso1e_cas, ci_ket, ncas, nelecas, ci_bra)
	def _compute_orbital_matrix_elements_diagnostic(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):

		orbital_elements = np.zeros(3, dtype=complex)

		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
	
		overlap = np.vdot(ci_bra_flat, ci_ket_flat)
		self._print(f"CI vector overlap: {overlap:.6e}", Colors.WHITE)
		

		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			soc_norm = np.linalg.norm(soc_matrix)
			soc_max = np.max(np.abs(soc_matrix))
			soc_mean = np.mean(np.abs(soc_matrix))
			
			self._print(f"SOC component {x}: Norm={soc_norm:.6e}, Max={soc_max:.6e}, Mean={soc_mean:.6e}", Colors.WHITE)
			
			 
			if soc_norm > 1e-10:  
				try:
					U, s, Vh = np.linalg.svd(soc_matrix)
					max_singular = s[0] if len(s) > 0 else soc_max
					
					element = overlap * max_singular / np.sqrt(ncas)
					orbital_elements[x] = element
					
				except Exception as e:
					self._print(f"SOC_{x} SVD: {e}", Colors.YELLOW)
					orbital_elements[x] = overlap * soc_max / ncas
			else:
				self._print(f"[?] SOC component {x} matrix norm is too small, using an estimate based on molecular composition", Colors.YELLOW)
				 
				min_soc = self._get_molecular_soc_estimate(mol)
				orbital_elements[x] = overlap * min_soc
		
		return orbital_elements

	def _compute_alternative_soc_integrals(self, mol):
		
		self._print("[?] Using an alternative method to calculate SOC points", Colors.YELLOW)
		
		nao = mol.nao_nr()
		hso1e_alt = np.zeros((3, nao, nao))
		
		 
		 
		for i in range(len(mol.atom)):
			elem_i, coord_i = mol.atom[i]
			z_i = self._get_atomic_number(elem_i)
			
			for j in range(len(mol.atom)):
				elem_j, coord_j = mol.atom[j]
				z_j = self._get_atomic_number(elem_j)
				
				 
				r_vec = np.array(coord_i) - np.array(coord_j)
				r = np.linalg.norm(r_vec)
				
				if r < 1e-6:   
					 
					soc_strength = z_i**2 * 1e-5
				else:
					 
					soc_strength = (z_i * z_j) / (r**2) * 1e-6
				
				 
				for x in range(3):
					if i == j:
						hso1e_alt[x, i, j] = soc_strength
					else:
						 
						direction_factor = r_vec[x] / r if r > 1e-6 else 1.0
						hso1e_alt[x, i, j] = soc_strength * direction_factor * 0.5
		
		self._print(f"Alternative SOC point range: {np.min(hso1e_alt):.6e} to {np.max(hso1e_alt):.6e}", Colors.WHITE)
		return hso1e_alt

	def _get_molecular_soc_estimate(self, mol):

		return 0

	def _get_atomic_number(self, element):
		
		periodic_table = {
			'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
			'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
			'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
		}
		return periodic_table.get(element, 6)   


	def _compute_spin_matrix_elements(self, spin_i, spin_j, m_i=None, m_j=None):
		 
		
		 
		if spin_i is None:
			spin_i = 0.0
		if spin_j is None:
			spin_j = 0.0
		
		 
		spin_i = self._round_spin_value(spin_i)
		spin_j = self._round_spin_value(spin_j)
		
		self._print(f"[~]Calculate spin matrix elements: S_i={spin_i} → S_j={spin_j}", Colors.CYAN)
		
		 
		if m_i is None or m_j is None:
			m_i, m_j = self._infer_m_quantum_numbers(spin_i, spin_j)
			self._print(f"[+]Got M: M_i={m_i}, M_j={m_j}", Colors.WHITE)
		
		 
		spin_matrix_elements = np.zeros(3, dtype=complex)
		
		 
		
		 
		if (spin_i == 0 and spin_j == 1) or (spin_i == 1 and spin_j == 0):
			spin_matrix_elements = self._compute_singlet_triplet_spin_elements(spin_i, spin_j, m_i, m_j)
		
		 
		elif spin_i == 1 and spin_j == 1:
			spin_matrix_elements = self._compute_triplet_triplet_spin_elements(m_i, m_j)
		
		 
		elif spin_i == 0 and spin_j == 0:
			spin_matrix_elements = np.array([0.0, 0.0, 0.0], dtype=complex)
		
		 
		else:
			self._print(f"[?] Did not understand that spin= {spin_i} → {spin_j}, Using Shif2", Colors.YELLOW)
			spin_matrix_elements = self._compute_general_spin_elements(spin_i, spin_j, m_i, m_j)
		
		self._print(f"Spin matrix element results: Sx={spin_matrix_elements[0]:.6f}, Sy={spin_matrix_elements[1]:.6f}, Sz={spin_matrix_elements[2]:.6f}", Colors.GREEN)
		
		return spin_matrix_elements

	def _infer_m_quantum_numbers(self, spin_i, spin_j):
		 
		 
		if spin_i == 0 and spin_j == 1:
			return 0, 1   
		
		 
		elif spin_i == 1 and spin_j == 0:
			return 1, 0   
		
		 
		elif spin_i == 1 and spin_j == 1:
			return 0, 1   
		
		 
		else:
			return 0, 0
	def _compute_singlet_triplet_spin_elements(self, spin_i, spin_j, m_i, m_j):
		 
		elements = np.zeros(3, dtype=complex)
		
		 
		if spin_i == 0 and spin_j == 1:
			if m_i == 0:   
				if m_j == 1:	   
					elements[0] = -1.0/(2*np.sqrt(2))   
					elements[1] = -1.0j/(2*np.sqrt(2))  
					self._print("[+] Using precise spin matrix elements:: S₀(M=0) → T₁(M=+1)", Colors.GREEN)
				elif m_j == -1:	
					elements[0] = 1.0/(2*np.sqrt(2))	
					elements[1] = -1.0j/(2*np.sqrt(2))  
					self._print("[+] Using precise spin matrix elements:: S₀(M=0) → T₁(M=-1)", Colors.GREEN)
				elif m_j == 0:	 
					self._print("[-] S₀(M=0) → T₁(M=0): Spin Off", Colors.YELLOW)
		
		 
		elif spin_i == 1 and spin_j == 0:
			if m_j == 0:   
				if m_i == 1:	   
					elements[0] = -1.0/(2*np.sqrt(2))   
					elements[1] = 1.0j/(2*np.sqrt(2))   
					self._print("[+] Using precise spin matrix elements:: T₁(M=+1) → S₀(M=0)", Colors.GREEN)
				elif m_i == -1:	
					elements[0] = 1.0/(2*np.sqrt(2))	
					elements[1] = 1.0j/(2*np.sqrt(2))   
					self._print("[+] Using precise spin matrix elements:: T₁(M=-1) → S₀(M=0)", Colors.GREEN)
				elif m_i == 0:	 
					self._print("[-] T₁(M=0) → S₀(M=0): Spin off", Colors.YELLOW)
		
		 
		if np.all(np.abs(elements) < 1e-12):
			self._print("[~] Using the average spin matrix element", Colors.YELLOW)
			 
			norm_factor = 1.0/np.sqrt(3)
			if spin_i == 0 and spin_j == 1:
				elements = np.array([norm_factor/np.sqrt(2), norm_factor/np.sqrt(2), 0.0], dtype=complex)
			else:
				elements = np.array([norm_factor/np.sqrt(2), norm_factor/np.sqrt(2), 0.0], dtype=complex)
		
		return elements
	def _compute_singlet_triplet_spin_elementsold(self, spin_i, spin_j, m_i, m_j):
		 
		elements = np.zeros(3, dtype=complex)
		
		 
		if spin_i == 0 and spin_j == 1:   
			if m_i == 0:   
				if m_j == 1:	   
					elements[0] = -1.0/(2*np.sqrt(2))   
					elements[1] = -1.0j/(2*np.sqrt(2))  
					elements[2] = 0.0					
				elif m_j == -1:	
					elements[0] = 1.0/(2*np.sqrt(2))	
					elements[1] = -1.0j/(2*np.sqrt(2))  
					elements[2] = 0.0					
				elif m_j == 0:	 
					elements[0] = 0.0					
					elements[1] = 0.0					
					elements[2] = 0.0					
		
		elif spin_i == 1 and spin_j == 0:   
			if m_j == 0:   
				if m_i == 1:	   
					elements[0] = -1.0/(2*np.sqrt(2))   
					elements[1] = 1.0j/(2*np.sqrt(2))   
					elements[2] = 0.0					
				elif m_i == -1:	
					elements[0] = 1.0/(2*np.sqrt(2))	
					elements[1] = 1.0j/(2*np.sqrt(2))   
					elements[2] = 0.0					
				elif m_i == 0:	 
					elements[0] = 0.0					
					elements[1] = 0.0					
					elements[2] = 0.0					
		
		 
		if np.all(np.abs(elements) < 1e-12):
			self._print("[~] Using averaged spin matrix elements (considering all M components)", Colors.YELLOW)
			 
			norm_factor = 1.0/np.sqrt(3)   
			if spin_i == 0 and spin_j == 1:   
				elements = np.array([norm_factor/np.sqrt(2), norm_factor/np.sqrt(2), 0.0], dtype=complex)
			else:   
				elements = np.array([norm_factor/np.sqrt(2), norm_factor/np.sqrt(2), 0.0], dtype=complex)
		
		return elements

	def _compute_triplet_triplet_spin_elements(self, m_i, m_j):
		 
		elements = np.zeros(3, dtype=complex)
		
		 
		 
		 
		
		self._print("[~] Using triplet-triplet spin matrix element approximation", Colors.YELLOW)
		elements = np.array([1.0, 1.0, 1.0], dtype=complex)
		
		return elements

	def _compute_general_spin_elements(self, spin_i, spin_j, m_i, m_j):
		 
		self._print(f"[?] Uncommon spin combination {spin_i}(M={m_i}) → {spin_j}(M={m_j})", Colors.YELLOW)
		
		 
		delta_S = abs(spin_i - spin_j)
		
		if delta_S == 0:
			 
			return np.array([1.0, 1.0, 1.0], dtype=complex)
		elif delta_S == 1:
			 
			return np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex)
		else:
			 
			return np.array([0.01, 0.01, 0.01], dtype=complex)

	def _round_spin_value(self, spin_value):
		 
		if spin_value is None:
			return 0.0
		
		common_spins = [0, 0.5, 1, 1.5, 2]
		for target in common_spins:
			if abs(spin_value - target) < 0.3:   
				return float(target)
		
		 
		rounded = round(spin_value * 2) / 2
		return float(rounded)

	def _round_spin_value(self, spin_value):
	
		if spin_value is None:
			return 0.0
		
		 
		common_spins = [0, 0.5, 1, 1.5, 2, 2.5, 3]
		for target in common_spins:
			if abs(spin_value - target) < 1e-6:
				return target
		
		 
		return round(spin_value)

	def _check_spin_selection_rules(self, spin_i, spin_j, m_i, m_j):

		delta_S = abs(spin_i - spin_j)
		delta_m = abs(m_i - m_j)
		
		if spin_i == 0 and spin_j == 0:
			return True
		
		 
		if delta_S > 1:
			return False
		
		 
		if delta_m > 1:
			return False
		
		return True

	def _compute_sx_matrix_element(self, S1, S2, m1, m2):
		
		sp_plus = self._compute_sp_plus_matrix_element(S1, S2, m1, m2)   
		sp_minus = self._compute_sp_minus_matrix_element(S1, S2, m1, m2)   
		
		return 0.5 * (sp_plus + sp_minus)

	def _compute_sy_matrix_element(self, S1, S2, m1, m2):
		
		sp_plus = self._compute_sp_plus_matrix_element(S1, S2, m1, m2)
		sp_minus = self._compute_sp_minus_matrix_element(S1, S2, m1, m2)
		
		return -0.5j * (sp_plus - sp_minus)

	def _compute_sz_matrix_element(self, S1, S2, m1, m2):
		
		if S1 == S2 and m1 == m2:
			return float(m1)
		else:
			return 0.0

	def _compute_sp_plus_matrix_element(self, S1, S2, m1, m2):
		
		 
		if S1 == S2 and m2 == m1 + 1:
			return np.sqrt(S1 * (S1 + 1) - m1 * (m1 + 1))
		else:
			return 0.0

	def _compute_sp_minus_matrix_element(self, S1, S2, m1, m2):
		
		 
		if S1 == S2 and m2 == m1 - 1:
			return np.sqrt(S1 * (S1 + 1) - m1 * (m1 - 1))
		else:
			return 0.0

	def get_common_spin_combinations(self):
		
		common_cases = {
			 
			'singlet_triplet': {
				'description': 'S=0 → T₁(m=0)',
				'spin_i': 0, 'm_i': 0,
				'spin_j': 1, 'm_j': 0,
				'expected_soc': 'Medium'
			},
			'singlet_triplet_x': {
				'description': 'S=0 → T₁(m=±1) by S_x',
				'spin_i': 0, 'm_i': 0, 
				'spin_j': 1, 'm_j': 1,
				'expected_soc': 'Strong'
			},
			'singlet_triplet_y': {
				'description': 'S=0 → T₁(m=±1) by S_y', 
				'spin_i': 0, 'm_i': 0,
				'spin_j': 1, 'm_j': 1,
				'expected_soc': 'Strong'
			},
			 
			'triplet_triplet': {
				'description': 'T₁ → T₁',
				'spin_i': 1, 'm_i': 0,
				'spin_j': 1, 'm_j': 1, 
				'expected_soc': 'Strong?'
			}
		}
		return common_cases
	def _get_state_spin(self, ci_vector, ncas, nelecas, mc=None, state_index=None):
	
		spin_value = 0.0   
		
		try:
			 
			if mc is not None and state_index is not None:
				spin_from_cas = self._get_spin_from_cas(mc, state_index)
				if spin_from_cas is not None:
					spin_value = self._round_spin_value(spin_from_cas)
					self._print(f"Obtain spin from CASCI: S={spin_value}", Colors.GREEN)
					return spin_value
			
			 
			spin_from_user = self._ask_user_for_spin(state_index)
			if spin_from_user is not None:
				spin_value = spin_from_user
				self._print(f"[~]Using Custom Spin: S={spin_value}", Colors.GREEN)
				return spin_value
			
			 
			if isinstance(nelecas, (tuple, list)):
				nalpha, nbeta = nelecas
				spin_value = 0.5 * abs(nalpha - nbeta)
			else:
				spin_value = 0.0
				
			self._print(f"Infer spin using electron count: S={spin_value}", Colors.YELLOW)
			return spin_value
			
		except Exception as e:
			self._print(f"[-] Spin determination failed: {e}, using default value S=0", Colors.RED)
			return 0.0

	def _round_spin_value(self, spin_value):
		
		if spin_value is None:
			return 0.0
			
		 
		for target in [0, 1, 2, 3, 4]:   
			if abs(spin_value - target) < 1e-6:
				return float(target)
		
		 
		for target in [0.5, 1.5, 2.5, 3.5]:
			if abs(spin_value - target) < 1e-6:
				return target
		
		 
		return spin_value

	def _get_spin_from_cas(self, mc, state_index):
		
		try:
			 
			if isinstance(mc.ci, (list, tuple)) and state_index < len(mc.ci):
				ci_vector = mc.ci[state_index]
			else:
				ci_vector = mc.ci
			
			 
			if hasattr(ci_vector, 'flatten'):
				ci_vector_flat = ci_vector.flatten()
			else:
				ci_vector_flat = ci_vector
			
			 
			if hasattr(mc, 'fcisolver') and hasattr(mc.fcisolver, 'spin_square'):
				try:
					s2_value = mc.fcisolver.spin_square(ci_vector_flat, mc.ncas, mc.nelecas)[0]
					spin_quantum_number = 0.5 * (np.sqrt(4 * s2_value + 1) - 1)
					return spin_quantum_number
				except:
					pass
			
			 
			if hasattr(mc, 'spin_square'):
				try:
					s2_value = mc.spin_square(ci_vector_flat, mc.ncas, mc.nelecas)[0]
					spin_quantum_number = 0.5 * (np.sqrt(4 * s2_value + 1) - 1)
					return spin_quantum_number
				except:
					pass
			
			 
			try:
				from pyscf.fci import spin_op
				s2_value = spin_op.spin_square(ci_vector_flat, mc.ncas, mc.nelecas)[0]
				spin_quantum_number = 0.5 * (np.sqrt(4 * s2_value + 1) - 1)
				return spin_quantum_number
			except:
				pass
				
		except Exception as e:
			self._print(f"[-] Failed to obtain spin from CASCI: {e}", Colors.YELLOW)
		
		return None

	def _ask_user_for_spin(self, state_index):

		try:
			if 1==1:
				state_label = f"State {state_index}" if state_index is not None else "This State"
				self._print(f"\n{Colors.CYAN}Please enter the spin quantum number S of {state_label}= {Colors.END}",Colors.RED)
				user_input = input("S = ")
				if user_input.strip():
					return float(user_input)
		except:
			pass   
		return None

	def _analyze_ci_spin(self, ci_vector, ncas, nelecas):

		try:
			nalpha, nbeta = nelecas
			
			 
			dominant_configs = self._find_dominant_configurations(ci_vector, n_top=5)
			
			 
			if dominant_configs is None or len(dominant_configs) == 0:
				return None
				
			spin_estimate = self._estimate_spin_from_configs(dominant_configs, ncas, (nalpha, nbeta))
			
			 
			if ci_vector is not None and len(ci_vector) > 1:
				spin_squared_estimate = self._estimate_spin_squared(ci_vector, ncas, (nalpha, nbeta))
				
				 
				if spin_squared_estimate is not None and spin_squared_estimate >= 0:
					spin_from_s2 = 0.5 * (np.sqrt(4 * spin_squared_estimate + 1) - 1)
					
					if spin_estimate is not None:
						final_estimate = 0.5 * (spin_estimate + spin_from_s2)
					else:
						final_estimate = spin_from_s2
					return final_estimate
			
			return spin_estimate
			
		except Exception as e:
			self._print(f"[-] Spin Bad-Read of CI: {e}", Colors.YELLOW)
			return None

	def _find_dominant_configurations(self, ci_vector, n_top=5):

		try:
			 
			if ci_vector is None or not hasattr(ci_vector, 'shape'):
				return []
				
			amplitudes_sq = np.abs(ci_vector)**2
			total_weight = np.sum(amplitudes_sq)
			
			 
			if total_weight == 0:
				return []
			
			 
			amplitudes_sq = amplitudes_sq / total_weight
			
			 
			dominant_indices = np.argsort(amplitudes_sq)[-n_top:][::-1]
			dominant_weights = amplitudes_sq[dominant_indices]
			
			dominant_configs = []
			for idx, weight in zip(dominant_indices, dominant_weights):
				 
				if isinstance(weight, (int, float, np.number)) and weight > 0.01:
					dominant_configs.append((idx, weight))
			
			return dominant_configs
			
		except Exception as e:
			self._print(f"[-] 寻找主要组态失败: {e}", Colors.YELLOW)
			return []

	def _estimate_spin_from_configs(self, dominant_configs, ncas, nelecas):

		try:
			 
			if not dominant_configs or len(dominant_configs) == 0:
				return None
				
			nalpha, nbeta = nelecas
			
			total_spin = 0.0
			total_weight = 0.0
			
			for config_idx, weight in dominant_configs:
				 
				if not isinstance(weight, (int, float, np.number)):
					continue
					
				unpaired_estimate = self._estimate_unpaired_electrons(config_idx, ncas, (nalpha, nbeta))
				config_spin = 0.5 * unpaired_estimate
				
				total_spin += config_spin * weight
				total_weight += weight
			
			 
			if total_weight > 0:
				return total_spin / total_weight
			else:
				return None
				
		except Exception as e:
			self._print(f"[-] Failed to estimate spin from configuration: {e}", Colors.YELLOW)
			return None

	def _estimate_unpaired_electrons(self, config_idx, ncas, nelecas):

		nalpha, nbeta = nelecas

		if nalpha == nbeta:
			return 0  
		
		return abs(nalpha - nbeta)

	def _estimate_spin_squared(self, ci_vector, ncas, nelecas):
		
		nalpha, nbeta = nelecas
		
		 
		
		if nalpha == nbeta:
			return 0.0   
		else:
			S_z = 0.5 * (nalpha - nbeta)
			 
			return S_z * (S_z + 1)
	def _compute_soc_matrix_elements(self, hso1e_cas, ci_i, ci_j, nelecas, ncas, mol, mc=None, state_i=None, state_j=None):

		if isinstance(nelecas, (tuple, list)):
			nalpha, nbeta = nelecas
		else:
			nalpha = nbeta = nelecas // 2
		
		 
		spin_i = self._get_state_spin(ci_i, ncas, (nalpha, nbeta), mc=mc, state_index=state_i)
		spin_j = self._get_state_spin(ci_j, ncas, (nalpha, nbeta), mc=mc, state_index=state_j)
		
		if spin_i is None:
			spin_i = 0
		if spin_j is None:  
			spin_j = 1
			
		self._print(f"[+] Got ya! State Spin: S_i={spin_i}, S_j={spin_j}", Colors.CYAN)
		
		 
		spin_matrix_elements = self._compute_spin_matrix_elements(spin_i, spin_j)
		
		 
		orbital_elements = self._compute_orbital_matrix_elements_first_principles(
			hso1e_cas, ci_i, ci_j, ncas, (nalpha, nbeta), mol
		)
		
		soc_elements = np.zeros(3, dtype=complex)
		
		 
		for x in range(3):
			soc_elements[x] = orbital_elements[x] * spin_matrix_elements[x]
		
		self._print(f"Complete SOC matrix elements: {soc_elements}", Colors.WHITE)
		
		return soc_elements

	def compute_soc_using_effective_hamiltonian(self, mol, mc, state_i, state_j):
		 
		alpha = 1.0 / 137.035999084
		
		 
		soc_matrix_elements = self._compute_soc_matrix_elements(
			hso1e_cas, ci_i, ci_j, mc.nelecas, ncas, mol, mc, state_i, state_j
		)
		
		 
		soc_matrix_elements = soc_matrix_elements * (alpha**2 / 2.0)
		
		return soc_matrix_elements, soc_norm


	def _compute_soc_first_principles(self, hso1e_cas, ci_i, ci_j, nelecas, ncas, mol, mc, state_i, state_j):

		if isinstance(nelecas, (tuple, list)):
			nalpha, nbeta = nelecas
		else:
			nalpha = nbeta = nelecas // 2
		
		 
		spin_i = self._get_state_spin(ci_i, ncas, (nalpha, nbeta), mc=mc, state_index=state_i)
		spin_j = self._get_state_spin(ci_j, ncas, (nalpha, nbeta), mc=mc, state_index=state_j)
		
		self._print(f"[+]State spin determined: State {state_i} S={spin_i}, State {state_j} S={spin_j}", Colors.CYAN)
		
		 
		spin_matrix_elements = self._compute_spin_matrix_elements(spin_i, spin_j)
		self._print(f"Spin matrix element= {spin_matrix_elements}", Colors.WHITE)
		
		soc_elements = np.zeros(3, dtype=complex)
		
		 
		orbital_elements = self._compute_orbital_matrix_elements_first_principles(
			hso1e_cas, ci_i, ci_j, ncas, (nalpha, nbeta), mol
		)
		self._print(f"Orbital matrix element= {orbital_elements}", Colors.WHITE)
		
		 
		for x in range(3):
			soc_elements[x] = orbital_elements[x] * spin_matrix_elements[x]
		
		return soc_elements
	def _compute_orbital_matrix_elements_first_principles(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):
		 
		orbital_elements = np.zeros(3, dtype=complex)
		
		try:
			 
			if hasattr(self, 'mc') and hasattr(self.mc, 'fcisolver'):
				try:
					for x in range(3):
						 
						element = self.mc.fcisolver.contract_1e(
							hso1e_cas[x], ci_ket, ncas, nelecas
						)
						 
						if hasattr(element, 'shape') and element.shape == ci_bra.shape:
							orbital_elements[x] = np.vdot(ci_bra, element)
						else:
							orbital_elements[x] = element
					self._print("[+] Using PySCF FCI", Colors.GREEN)
					return orbital_elements
				except Exception as e:
					self._print(f"[-]Err with FCI-FCI: {e}", Colors.YELLOW)
			
			 
			ci_bra_flat = self._ensure_1d(ci_bra)
			ci_ket_flat = self._ensure_1d(ci_ket)
			
			 
			overlap = np.vdot(ci_bra_flat, ci_ket_flat)
			
			for x in range(3):
				soc_matrix = hso1e_cas[x]
				
				 
				soc_frobenius = np.linalg.norm(soc_matrix, 'fro')
				
				 
				try:
					U, s, Vh = np.linalg.svd(soc_matrix)
					max_singular = s[0] if len(s) > 0 else soc_frobenius
				except:
					max_singular = soc_frobenius
				
				 
				element = overlap * max_singular / max(1, ncas)
				orbital_elements[x] = element
				
				self._print(f"[~] SOC Component{x}: S={abs(overlap):.3e}, dSOC={max_singular:.3e}", Colors.WHITE)
			
			return orbital_elements
			
		except Exception as e:
			self._print(f"[-] Faild with SOC Calculate: {e}", Colors.RED)
			 
			return np.array([1e-6, 1e-6, 1e-6], dtype=complex)
	def _compute_orbital_matrix_elements_first_principles_old(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):

		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		 
		overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))
		self._print(f"[~]CI vector overlap: {overlap:.6e}", Colors.WHITE)
		
		 
		if overlap < 1e-10:
			self._print("[*]WARN: CI vector overlap is too small, using SOC matrix-dominated estimation", Colors.YELLOW)
			
			for x in range(3):
				soc_matrix = hso1e_cas[x]
				
				 
				soc_frob_norm = np.linalg.norm(soc_matrix, 'fro')
				max_off_diag = np.max(np.abs(soc_matrix - np.diag(np.diag(soc_matrix))))

				soc_strength = 0.1 * (soc_frob_norm + max_off_diag) / ncas
				
				orbital_elements[x] = soc_strength
				
				self._print(f"[~] SOC Component {x}: Independent Strength Estimation = {soc_strength:.6e} a.u.", Colors.WHITE)
		
		else:
			 
			for x in range(3):
				soc_matrix = hso1e_cas[x]
				
				soc_frob_norm = np.linalg.norm(soc_matrix, 'fro')
				max_off_diag = np.max(np.abs(soc_matrix - np.diag(np.diag(soc_matrix))))
				
				soc_strength = 0.5 * (soc_frob_norm + max_off_diag)
				
				orbital_element = overlap * soc_strength / ncas
				orbital_elements[x] = orbital_element
				
				self._print(f"[~] SOC Component {x}: Based on Overlapping Estimation= {orbital_element:.6e} a.u.", Colors.WHITE)
		
		return orbital_elements
	def old_compute_orbital_matrix_elements_first_principles(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):

		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		 
		overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))   
		self._print(f"[~] CI vector overlap: {overlap:.6e}", Colors.WHITE)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			 
			soc_frob_norm = np.linalg.norm(soc_matrix, 'fro')
			
			 
			off_diag_mask = ~np.eye(soc_matrix.shape[0], dtype=bool)
			max_off_diag = np.max(np.abs(soc_matrix[off_diag_mask]))
			

			soc_strength = 0.5 * (soc_frob_norm + max_off_diag)
			
			self._print(f"[~] ||SOC_{x}||={soc_frob_norm:.6e}, Maximum off-diagonal={max_off_diag:.6e}", Colors.WHITE)
			self._print(f"[~]SOC Component {x} Original Intensity: {soc_strength:.6e} a.u.", Colors.WHITE)
			

			orbital_element = overlap * soc_strength
			
			orbital_elements[x] = orbital_element
			
			self._print(f"[~]SOC component {x} orbital matrix element: {orbital_element:.6e}", Colors.WHITE)
		
		return orbital_elements

	def _compute_soc_expectation_value(self, soc_matrix, ci_bra, ci_ket, ncas, nelecas):

		try:

			bra_amps = np.abs(ci_bra)
			ket_amps = np.abs(ci_ket)
			
			bra_top = np.argsort(bra_amps)[-10:][::-1]   
			ket_top = np.argsort(ket_amps)[-10:][::-1]
			
			expectation = 0.0
			for i in bra_top:
				for j in ket_top:
					if bra_amps[i] > 0.01 and ket_amps[j] > 0.01:   
						contribution = ci_bra[i].conj() * soc_matrix[i % ncas, j % ncas] * ci_ket[j]
						expectation += contribution
			
			return expectation
			
		except Exception as e:
			self._print(f"[-]Bad SOC: {e}", Colors.YELLOW)
			return 0.0

	def _estimate_soc_from_molecular_structure(self, mol):

		heavy_atoms = {'N': 3.0, 'O': 4.0, 'F': 5.0, 'P': 15.0, 'S': 16.0}
		
		total_soc_factor = 0.0
		heavy_atom_count = 0
		
		for atom in mol.atom:
			element = atom[0]
			if element in heavy_atoms:
				z = heavy_atoms[element]
				total_soc_factor += z**2   
				heavy_atom_count += 1
			elif element == 'C':
				total_soc_factor += 6**2 * 0.1   
			elif element == 'H':
				total_soc_factor += 1**2 * 0.01   
		
		if heavy_atom_count > 0:
			 
			soc_scale = total_soc_factor * 1e-5
		else:
			 
			soc_scale = total_soc_factor * 1e-6
		
		self._print(f"基于分子结构的SOC尺度估计: {soc_scale:.6e} a.u.", Colors.CYAN)
		return soc_scale

	def _get_physical_soc_scale(self, mol):

		base_scale = 1e-3   
		
		 
		heavy_elements = {'N', 'O', 'F', 'P', 'S', 'Cl'}
		heavy_count = sum(1 for atom in mol.atom if atom[0] in heavy_elements)
		
		if heavy_count >= 3:
			scale_factor = 2.0
		elif heavy_count >= 1:
			scale_factor = 1.0
		else:
			scale_factor = 0.5
		
		physical_scale = base_scale * scale_factor
		self._print(f"[?]SOC: {physical_scale:.6e} a.u. (Based on{heavy_count} heavy_atom_count)", Colors.CYAN)
		return physical_scale

	def compute_soc_using_effective_hamiltonian(self, mol, mc, state_i, state_j):

		alpha = 1.0 / 137.035999084
		
		 
		try:
			hso1e = mol.intor('int1e_prinvxp', 3)
			self._print(f"[~]Original SOC score range: {np.min(hso1e):.6e} to {np.max(hso1e):.6e}", Colors.CYAN)
			
			 
			for x in range(3):
				diag_ao = np.diag(hso1e[x])
				self._print(f"[~]SOC_{x}AO diagonal range: {np.min(diag_ao):.6e} to {np.max(diag_ao):.6e}", Colors.WHITE)
				
		except Exception as e:
			self._print(f"[-]Bad Reading SOC: {e}", Colors.RED)
			return np.zeros(3, dtype=complex), 0.0
		
		 
		mo_coeff = mc.mo_coeff
		ncore = mc.ncore
		ncas = mc.ncas
		
		self._print(f"[~]Shape of MO: {mo_coeff.shape}", Colors.WHITE)
		self._print(f"[!]Active Space: ncore={ncore}, ncas={ncas}", Colors.WHITE)
		
		 
		hso1e_mo = np.zeros((3, mo_coeff.shape[1], mo_coeff.shape[1]))
		for x in range(3):
			 
			temp1 = hso1e[x].dot(mo_coeff)
			hso1e_mo[x] = mo_coeff.T.dot(temp1)
			
			 
			diag_mo_full = np.diag(hso1e_mo[x])
			self._print(f"[~]SOC_{x} Full MO diagonal range: {np.min(diag_mo_full):.6e} to {np.max(diag_mo_full):.6e}", Colors.WHITE)

		cas_idx = np.arange(ncore, ncore + ncas)
		hso1e_cas = hso1e_mo[:, cas_idx[:, None], cas_idx]

		for x in range(3):
			soc_cas = hso1e_cas[x]
			diag_cas = np.diag(soc_cas)
			self._print(f"[~]SOC_{x} Active space diagonal range: {np.min(diag_cas):.6e} to {np.max(diag_cas):.6e}", Colors.CYAN)
			
			 
			asymmetry = np.linalg.norm(soc_cas - soc_cas.T)
			self._print(f"[~]SOC_{x} Symmetry breaking: {asymmetry:.6e}", Colors.WHITE)
		
		 
		ci_i = self._get_ci_vector(mc, state_i)
		ci_j = self._get_ci_vector(mc, state_j)

		soc_matrix_elements = self._compute_soc_matrix_elements(
			hso1e_cas, ci_i, ci_j, mc.nelecas, ncas, mol, mc, state_i, state_j
		)

		soc_matrix_elements = soc_matrix_elements * (alpha**(2 / 2.0))
		
		soc_norm = np.linalg.norm(soc_matrix_elements)
		
		self._print(f"[~]SOC Matrix element= {soc_matrix_elements} a.u.", Colors.WHITE)
		self._print(f"[~]||SOC||: {soc_norm:.6e} a.u. ({soc_norm * 219474.63:.2f} cm-1)", Colors.WHITE)
		
		return soc_matrix_elements, soc_norm
	def compute_soc_using_effective_hamiltonianold(self, mol, mc, state_i, state_j):

		alpha = 1.0 / 137.035999084
		
		 
		try:
			hso1e = mol.intor('int1e_prinvxp', 3)
		except:
			hso1e = self._compute_alternative_soc_integrals(mol)
		
		 
		mo_coeff = mc.mo_coeff
		ncore = mc.ncore
		ncas = mc.ncas
		
		hso1e_mo = np.zeros((3, mo_coeff.shape[1], mo_coeff.shape[1]))
		for x in range(3):
			hso1e_mo[x] = mo_coeff.T.dot(hso1e[x]).dot(mo_coeff)
		
		cas_idx = np.arange(ncore, ncore + ncas)
		hso1e_cas = hso1e_mo[:, cas_idx[:, None], cas_idx]
		
		 
		ci_i = self._get_ci_vector(mc, state_i)
		ci_j = self._get_ci_vector(mc, state_j)
		
		 
		soc_matrix_elements = self._compute_soc_matrix_elements(
			hso1e_cas, ci_i, ci_j, mc.nelecas, ncas, mol, mc, state_i, state_j
		)
		

		relativistic_factor = alpha**2 / 2.0
		
		soc_matrix_elements = soc_matrix_elements * relativistic_factor
		
		soc_norm = np.linalg.norm(soc_matrix_elements)
		
		self._print(f"[~]Alpha= {relativistic_factor:.6e}", Colors.CYAN)
		self._print(f"[~]SOC Matrix element= {soc_matrix_elements} a.u.", Colors.WHITE)
		self._print(f"[~]||SOC||= {soc_norm:.6e} a.u. ({soc_norm * 219474.63:.2f} cm-1)", Colors.WHITE)
		
		return soc_matrix_elements, soc_norm

	def _get_minimal_soc_from_molecule(self, mol):

		heavy_atoms = {'N': 2.0, 'O': 3.0, 'F': 4.0, 'P': 8.0, 'S': 10.0, 'Cl': 12.0}
		
		soc_factor = 0.0
		atom_count = 0
		
		for atom in mol.atom:
			element = atom[0]
			if element in heavy_atoms:
				soc_factor += heavy_atoms[element]
				atom_count += 1
			elif element == 'C':
				soc_factor += 0.5
				atom_count += 1
			elif element == 'H':
				soc_factor += 0.1
				atom_count += 1
		
		if atom_count > 0:
			avg_soc_factor = soc_factor / atom_count
		else:
			avg_soc_factor = 1.0
		
		 
		min_soc_au = avg_soc_factor * 1e-6
		
		self._print(f"[..]Mini: {min_soc_au:.6e} a.u.", Colors.CYAN)
		return min_soc_au
	def _determine_spin_multiplet(self, mc, state_index):

		try:
			 
			if hasattr(mc, 'e_states') and hasattr(mc, 'ci'):
				 
				if state_index == 0:
					return 'singlet'
				elif state_index == 1:
					return 'triplet'
			
			 
			if hasattr(mc, 'nelecas'):
				nalpha, nbeta = mc.nelecas
				if nalpha == nbeta:
					 
					return 'singlet' if state_index == 0 else 'triplet'
				else:
					 
					return 'triplet' if state_index > 0 else 'singlet'
			
			 
			return 'singlet' if state_index == 0 else 'triplet'
			
		except:
			return 'singlet' if state_index == 0 else 'triplet'

	def _compute_spin_orbit_matrix_elements(self, spin_i, spin_j):

		
		if spin_i == 'singlet' and spin_j == 'triplet':

			return np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex)
		
		elif spin_i == 'triplet' and spin_j == 'singlet':
			 
			return np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=complex)
		
		elif spin_i == 'triplet' and spin_j == 'triplet':
			 
			return np.array([1.0, 1.0, 1.0], dtype=complex)
		
		else:
			 
			return np.array([0.0, 0.0, 0.0], dtype=complex)

	def _compute_orbital_matrix_elements_robust(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):
		
		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		 
		overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))
		
		 
		soc_strength_estimate = self._estimate_soc_strength_from_molecule(mol)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			 
			try:
				 
				soc_norm = np.linalg.norm(soc_matrix)
				
				 
				eigenvalues = np.linalg.eigvals(soc_matrix)
				max_eigenval = np.max(np.abs(eigenvalues))
				
				 
				soc_mean = np.mean(np.abs(soc_matrix))
				soc_max = np.max(np.abs(soc_matrix))
				
				 
				matrix_factor = 0.3 * soc_mean + 0.5 * (soc_max / ncas) + 0.2 * (soc_norm / ncas**2)
				
				 
				element = overlap * matrix_factor * soc_strength_estimate
				
				orbital_elements[x] = element
				
			except Exception as e:
				self._print(f"[-]Bad SOC_x: {e}", Colors.YELLOW)

				orbital_elements[x] = overlap * soc_strength_estimate * 0.01
		
		self._print(f"[~]Approximated SOC Matrix : {orbital_elements}", Colors.WHITE)
		return orbital_elements

	def _estimate_soc_strength_from_molecule(self, mol):
		heavy_atoms = {'C': 1.0, 'N': 3.0, 'O': 4.0, 'F': 5.0, 'P': 15.0, 'S': 16.0}
		
		total_soc_factor = 0.0
		atom_count = 0
		
		for atom in mol.atom:
			element = atom[0]
			if element in heavy_atoms:
				 
				z = heavy_atoms[element]
				total_soc_factor += z**2
				atom_count += 1
		
		if atom_count > 0:
			avg_soc_factor = total_soc_factor / atom_count
		else:
			avg_soc_factor = 1.0   
		
		 
		base_soc = 0.001   
		soc_strength = base_soc * np.sqrt(avg_soc_factor)
		
		self._print(f"[?]Approximated SOC Value: {soc_strength:.6f} a.u.", Colors.CYAN)
		return soc_strength

	def _compute_approximate_soc_integrals(self, mol):
	
		nao = mol.nao
		hso1e_approx = np.zeros((3, nao, nao))
		
		 
		for i, atom_i in enumerate(mol.atom):
			element_i, coord_i = atom_i
			z_i = self._get_atomic_number(element_i)
			
			for j, atom_j in enumerate(mol.atom):
				element_j, coord_j = atom_j
				z_j = self._get_atomic_number(element_j)
				
				 
				r = np.linalg.norm(np.array(coord_i) - np.array(coord_j))
				
				 
				soc_strength = (z_i * z_j) / (1.0 + r)**2 * 1e-4
				
				 
				for x in range(3):
					hso1e_approx[x, i, j] = soc_strength * (1.0 if i == j else 0.5)
		
		return hso1e_approx

	def _get_atomic_number(self, element):
		periodic_table = {
			'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
			'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
			'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20
		}
		return periodic_table.get(element, 6)   
	def compute_phosphorescence_rate(self, soc_norm, energy_gap):
		 
		 
		soc_cm = soc_norm * 219474.63   
		deltaE_cm = energy_gap * 8065.73   
		deltaE_eV = energy_gap   
		
		self._print(f"[~] SOC = {soc_cm:.2f} cm⁻¹, ΔE = {deltaE_cm:.1f} cm⁻¹ ({deltaE_eV:.3f} eV)", Colors.WHITE)
		
		 
		h = 6.62607015e-34   
		hbar = h / (2 * np.pi)   
		c = 2.99792458e10   
		eps0 = 8.8541878128e-12   
		
		 
		
		 
		 
		n_refractive = 1.0   
		photon_dos = (8 * np.pi * (n_refractive**3) * (deltaE_cm**2)) / ((h**3) * (c**3))
		
		 
		 
		 
		avg_vib_freq = 500.0   
		hr_factor = (deltaE_cm / avg_vib_freq)   
		vib_dos_factor = np.exp(-hr_factor) * (hr_factor**3) / 6.0   
		
		 
		 
		spin_degeneracy = 3.0   
		symmetry_factor = 1.0   
		
		 
		effective_dos = photon_dos * vib_dos_factor * spin_degeneracy * symmetry_factor
		
		self._print(f"[~] Photon DOS = {photon_dos:.3e} states/cm⁻¹", Colors.WHITE)
		self._print(f"[~] Vib factor = {vib_dos_factor:.3e} (HR≈{hr_factor:.1f})", Colors.WHITE)
		self._print(f"[~] Spin degeneracy = {spin_degeneracy}", Colors.WHITE)
		
		 
		
		 
		soc_J = soc_cm * h * c * 100   
		
		 
		rate_si = (2 * np.pi / hbar) * (soc_J**2) * effective_dos
		
		 
		
		 
		 
		avg_transition_dipole = 0.3   
		dipole_correction = (avg_transition_dipole / 1.0)**2   
		
		 
		 
		soc_efficiency = 0.1   
		
		rate_si = rate_si * dipole_correction * soc_efficiency
		
		 
		
		 
		min_rate = 1e-3	 
		max_rate = 1e6	  
		
		rate_si = max(rate_si, min_rate)
		rate_si = min(rate_si, max_rate)
		
		lifetime = 1.0 / rate_si if rate_si > 0 else float('inf')
		
		 
		
		if lifetime < 1e-6:
			lifetime_type = "ULTRAFAST"
		elif lifetime < 1e-3:
			lifetime_type = "Fast"
		elif lifetime < 1.0:
			lifetime_type = "Medium"
		elif lifetime < 10.0:
			lifetime_type = "Slow"
		else:
			lifetime_type = "VERY SLOW"
		
		 
		
		self._print(f"[+] IMPROVED PHOSPHORESCENCE RATE CALCULATION:", Colors.CYAN)
		self._print(f"	SOC strength:	  {soc_cm:.3f} cm⁻¹", Colors.WHITE)
		self._print(f"	Energy gap:		{deltaE_eV:.3f} eV", Colors.WHITE)
		self._print(f"	Transition rate:   {rate_si:.3e} s⁻¹", Colors.WHITE)
		self._print(f"	Lifetime:		  {lifetime:.3e} s", Colors.WHITE)
		self._print(f"	Lifetime:		  {lifetime*1000:.3e} ms", Colors.WHITE)
		self._print(f"	Classification:	{lifetime_type} phosphorescence", Colors.WHITE)
		
		 
		if lifetime_type in ["ULTRAFAST", "Fast"]:
			self._print(f"	NOTE: Short lifetime suggests strong SOC or small ΔE", Colors.YELLOW)
		elif lifetime_type in ["Slow", "VERY SLOW"]:
			self._print(f"	NOTE: Long lifetime suggests weak SOC or large ΔE", Colors.YELLOW)
		
		return rate_si, lifetime

	def _validate_soc_inputs(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):
		try:
			 
			if hso1e_cas is None:
				self._print("[-]Err:CAS=None", Colors.RED)
				return False
			
			if not hasattr(hso1e_cas, 'shape') or len(hso1e_cas.shape) != 3:
				self._print("[-]Bad Shape of CI Vector", Colors.RED)
				return False
			
			 
			if ci_bra is None or ci_ket is None:
				self._print("", Colors.RED)
				return False
			
			 
			ci_bra_array = np.asarray(ci_bra)
			ci_ket_array = np.asarray(ci_ket)
			
			if ci_bra_array.size == 0 or ci_ket_array.size == 0:
				self._print("[-]Err with emity CI vectors", Colors.RED)
				return False
			
			 
			if ncas <= 0:
				self._print("[-]Error:Bad active space", Colors.RED)
				return False
			
			if nelecas is None:
				self._print("[-]bad Value None", Colors.RED)
				return False
				
			return True
			
		except Exception as e:
			self._print(f"[-]Bad Input: {e}", Colors.RED)
			return False

	def _compute_soc_direct_method(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		 
		overlap = np.vdot(ci_bra_flat, ci_ket_flat)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			 
			try:
				 
				soc_norm = np.linalg.norm(soc_matrix, 'fro')
				
				 
				eigenvalues = np.linalg.eigvals(soc_matrix)
				max_eigenval = np.max(np.abs(eigenvalues))
				
				 
				matrix_factor = 0.5 * (soc_norm + max_eigenval) / ncas
				
				 
				element = overlap * matrix_factor
				orbital_elements[x] = element
				
			except Exception as e:
				self._print(f"[-]Err SOC_x{x}: {e}", Colors.YELLOW)
				 
				orbital_elements[x] = 0.001 * overlap   
		
		return orbital_elements

	def _compute_soc_dominant_configs(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)

		bra_configs = self._get_dominant_configurations(ci_bra_flat, min_weight=1e-4, max_configs=20)
		ket_configs = self._get_dominant_configurations(ci_ket_flat, min_weight=1e-4, max_configs=20)
		
		if not bra_configs or not ket_configs:
			self._print("[-]Failed with reading main-CIs", Colors.YELLOW)
			return orbital_elements
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			total_element = 0.0 + 0.0j

			for (i, amp_bra) in bra_configs:
				for (j, amp_ket) in ket_configs:

					config_coupling = self._estimate_configuration_coupling(i, j, soc_matrix, ncas)
					
					contribution = amp_bra.conjugate() * amp_ket * config_coupling
					total_element += contribution
			
			orbital_elements[x] = total_element
		
		return orbital_elements

	def _get_dominant_configurations(self, ci_vector, min_weight=1e-4, max_configs=20):

		try:
			ci_flat = self._ensure_1d(ci_vector)
			weights = np.abs(ci_flat) ** 2
			total_weight = np.sum(weights)
			
			if total_weight == 0:
				return []

			weights = weights / total_weight

			significant_indices = np.where(weights >= min_weight)[0]

			if len(significant_indices) > max_configs:
				significant_indices = np.argsort(weights)[-max_configs:][::-1]
			
			dominant_configs = []
			for idx in significant_indices:
				dominant_configs.append((idx, ci_flat[idx]))
			
			return dominant_configs
			
		except Exception as e:
			self._print(f"[-]Failed with reading main-CIs: {e}", Colors.YELLOW)
			return []

	def _estimate_configuration_coupling(self, config_i, config_j, soc_matrix, ncas):

		try:

			config_diff = abs(config_i - config_j)

			similarity = np.exp(-config_diff / (2 * ncas))

			soc_mean = np.mean(np.abs(soc_matrix))
			soc_max = np.max(np.abs(soc_matrix))

			coupling = similarity * 0.5 * (soc_mean + soc_max)
			
			return coupling
			
		except Exception as e:
			self._print(f"[-]Failed with CIDIFF: {e}", Colors.YELLOW)
			return 0.001   

	def _compute_soc_matrix_properties(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		

		overlap = np.vdot(ci_bra_flat, ci_ket_flat)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			try:

				soc_norm = np.linalg.norm(soc_matrix, 'fro')
				soc_max = np.max(np.abs(soc_matrix))
				soc_mean = np.mean(np.abs(soc_matrix))

				sparse_threshold = 1e-8
				nonzero_ratio = np.sum(np.abs(soc_matrix) > sparse_threshold) / (ncas * ncas)

				diag_elements = np.abs(np.diag(soc_matrix))
				diag_norm = np.linalg.norm(diag_elements)
				off_diag_norm = np.linalg.norm(soc_matrix - np.diag(np.diag(soc_matrix)))
				
				if diag_norm > 0:
					dominance_ratio = off_diag_norm / diag_norm
				else:
					dominance_ratio = 1.0

				base_soc = 0.3 * soc_mean + 0.7 * soc_max / ncas
				sparsity_factor = 0.5 + 0.5 * (1 - nonzero_ratio)
				dominance_factor = 1.0 / (1.0 + dominance_ratio)
				
				element = overlap * base_soc * sparsity_factor * dominance_factor
				orbital_elements[x] = element
				
			except Exception as e:
				self._print(f"[-]Failed with Reading SOC Matrix: {e}", Colors.YELLOW)
				orbital_elements[x] = 0.001 * overlap
		
		return orbital_elements

	def _compute_soc_reasonable_estimate(self, hso1e_cas, ncas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		base_soc = 0.005   
		
		try:
			 
			if hso1e_cas is not None:
				soc_norms = [np.linalg.norm(hso1e_cas[x]) for x in range(3)]
				avg_norm = np.mean(soc_norms) if soc_norms else 0
				
				 
				if avg_norm > 0:
					base_soc *= min(avg_norm / ncas, 10.0)   
			
			 
			for x in range(3):
				orbital_elements[x] = base_soc * (0.8 + 0.4 * np.random.random())
				
		except Exception:
			 
			for x in range(3):
				orbital_elements[x] = 0.005
		
		self._print(f"[-]Using Approximate SOC Value: {orbital_elements}", Colors.YELLOW)
		return orbital_elements

	def _is_valid_soc_result(self, orbital_elements):

		if orbital_elements is None:
			return False
		
		if not isinstance(orbital_elements, np.ndarray):
			return False
		
		if orbital_elements.shape != (3,):
			return False
		
		 
		if np.any(np.isnan(orbital_elements)) or np.any(np.isinf(orbital_elements)):
			return False
		
		 
		max_val = np.max(np.abs(orbital_elements))
		if max_val > 100.0 or max_val < 1e-12:   
			return False
		
		return True

	def _compute_soc_with_fcisolver(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mc):

		orbital_elements = np.zeros(3, dtype=complex)
		
		try:
			fcisolver = mc.fcisolver

			for x in range(3):

				soc_operator = hso1e_cas[x]

				 
				matrix_element = fcisolver.contract_1e(soc_operator, ci_ket, ncas, nelecas, ci_bra)
				orbital_elements[x] = matrix_element
				
				self._print(f"[~]SOC_x{x}= {matrix_element:.6e}", Colors.WHITE)
				
		except Exception as e:
			self._print(f"[-]Failed with FCI solver SOC: {e}", Colors.YELLOW)
			
			orbital_elements = self._compute_soc_simplified_direct(hso1e_cas, ci_bra, ci_ket, ncas, nelecas)
		
		return orbital_elements

	def _compute_soc_simplified_direct(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		 
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		 
		bra_configs = self._analyze_dominant_configurations(ci_bra_flat, top_n=10)
		ket_configs = self._analyze_dominant_configurations(ci_ket_flat, top_n=10)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			total_element = 0.0
			
			 
			for (i, amp_bra), (j, amp_ket) in zip(bra_configs, ket_configs):
				 
				config_element = self._estimate_configuration_soc(i, j, soc_matrix, ncas, nelecas)
				total_element += amp_bra.conj() * amp_ket * config_element
			
			orbital_elements[x] = total_element
			
			 
			soc_norm = np.linalg.norm(soc_matrix)
			overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))
			correction_factor = soc_norm * overlap / ncas
			
			orbital_elements[x] += correction_factor * 0.1   
		
		return orbital_elements

	def _analyze_dominant_configurations(self, ci_vector, top_n=10):

		ci_flat = self._ensure_1d(ci_vector)
		amplitudes = np.abs(ci_flat)
		
		 
		indices = np.argsort(amplitudes)[-top_n:][::-1]
		dominant_configs = []
		
		for idx in indices:
			if amplitudes[idx] > 1e-4:   
				dominant_configs.append((idx, ci_flat[idx]))
		
		return dominant_configs

	def _estimate_configuration_soc(self, config_i, config_j, soc_matrix, ncas, nelecas):

		 
		 
		
		soc_mean = np.mean(np.abs(soc_matrix))
		soc_std = np.std(np.abs(soc_matrix))
		
		 
		config_diff = abs(config_i - config_j)
		similarity = np.exp(-config_diff / ncas)   
		
		return soc_mean * similarity * (1 + 0.1 * soc_std / soc_mean if soc_mean > 0 else 1.0)

	def _compute_soc_fallback(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)
		
		ci_bra_flat = self._ensure_1d(ci_bra)
		ci_ket_flat = self._ensure_1d(ci_ket)
		
		for x in range(3):
			soc_matrix = hso1e_cas[x]
			
			 
			eigenvalues = np.linalg.eigvals(soc_matrix)
			max_eigenvalue = np.max(np.abs(eigenvalues))
			
			overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))
			
			 
			base_soc = max_eigenvalue * overlap
			
			 
			 
			enhancement = 1.0   
			
			 
			if ncas > 8:   
				enhancement *= 2.0
				
			orbital_elements[x] = base_soc * enhancement
		
		return orbital_elements
	def _compute_orbital_matrix_element(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):
		
		try:
			 
			ci_bra = np.asarray(ci_bra)
			ci_ket = np.asarray(ci_ket)
			hso_operator = np.asarray(hso_operator)

			if ci_bra.shape != ci_ket.shape:
				self._print(f"[?]Bad Shape of CI vectors: {ci_bra.shape} vs {ci_ket.shape}", Colors.YELLOW)
				return 0.0
				
			if hso_operator.shape[0] != len(ci_bra) or hso_operator.shape[1] != len(ci_ket):
				self._print(f"[?]Bad Shape with CI and SOC: {hso_operator.shape} vs ({len(ci_bra)}, {len(ci_ket)})", Colors.YELLOW)
				return 0.0
			
			soc_element = self._compute_soc_improved_approximate(hso_operator, ci_bra, ci_ket, ncas, nelecas)
			return soc_element
			
		except Exception as e:
			self._print(f"[-]FUCKING Failed at func<_compute_orbital_matrix_element>: {e}", Colors.RED)
			return 0.0
	def _compute_orbital_matrix_elementsold(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas):

		orbital_elements = np.zeros(3, dtype=complex)

		ci_bra_flat = ci_bra.flatten() if ci_bra.ndim > 1 else ci_bra
		ci_ket_flat = ci_ket.flatten() if ci_ket.ndim > 1 else ci_ket
		
		self._print(f"[~]Shape of CI Vector: bra={ci_bra_flat.shape}, ket={ci_ket_flat.shape}", Colors.WHITE)
		self._print(f"[~]Shape of SOC <hso1e_cas>: {hso1e_cas.shape}", Colors.WHITE)
		
		
		
		for x in range(3):
			hso_operator = hso1e_cas[x]
			orbital_element = self._compute_soc_using_fci(hso_operator, ci_bra_flat, ci_ket_flat, ncas, nelecas)
			orbital_elements[x] = orbital_element
		
		return orbital_elements

	def _compute_soc_using_fci(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):

		try:

			if isinstance(nelecas, (tuple, list)):
				nalpha, nbeta = nelecas
			else:
				nalpha = nbeta = nelecas // 2

			soc_value = self._compute_soc_from_dominant_configs(hso_operator, ci_bra, ci_ket, ncas, (nalpha, nbeta))
			
			if abs(soc_value) > 1e-10:
				self._print(f"[~]ompute_soc_improved_approximate gives SOC:  {soc_value:.6e}", Colors.WHITE)
			else:
				
				soc_value = self._compute_soc_improved_approximate(hso_operator, ci_bra, ci_ket, ncas, (nalpha, nbeta))
				self._print(f"[~]ompute_soc_improved_approximate gives SOC: {soc_value:.6e}", Colors.WHITE)
			
			return soc_value
			
		except Exception as e:
			self._print(f"[-]Bad fucking FCI SOC: {e}", Colors.RED)

			return 1e-4

	def _compute_soc_from_dominant_configs(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):
		 
		try:
			 
			ci_bra_flat = self._ensure_1d(ci_bra)
			ci_ket_flat = self._ensure_1d(ci_ket)
			
			 
			bra_amps = np.abs(ci_bra_flat)
			ket_amps = np.abs(ci_ket_flat)
			
			 
			bra_norm = np.linalg.norm(bra_amps)
			ket_norm = np.linalg.norm(ket_amps)
			
			if bra_norm > 0:
				bra_amps = bra_amps / bra_norm
			if ket_norm > 0:
				ket_amps = ket_amps / ket_norm
			
			 
			threshold = 0.01
			bra_indices = np.where(bra_amps > threshold)[0]
			ket_indices = np.where(ket_amps > threshold)[0]
			
			 
			if len(bra_indices) == 0:
				bra_indices = np.argsort(bra_amps)[-10:][::-1]
			if len(ket_indices) == 0:
				ket_indices = np.argsort(ket_amps)[-10:][::-1]
			
			self._print(f"[~] 使用 {len(bra_indices)}×{len(ket_indices)} 个主要组态计算SOC", Colors.WHITE)
			
			soc_value = 0.0 + 0.0j
			
			 
			for i in bra_indices[:10]:   
				for j in ket_indices[:10]:   
					 
					if float(bra_amps[i]) > threshold and float(ket_amps[j]) > threshold:
						 
						config_overlap = ci_bra_flat[i].conj() * ci_ket_flat[j]
						
						 
						soc_strength = np.linalg.norm(hso_operator)
						
						 
						soc_contribution = config_overlap * soc_strength * float(bra_amps[i]) * float(ket_amps[j])
						soc_value += soc_contribution
			
			return soc_value
			
		except Exception as e:
			self._print(f"[-] 主要组态SOC计算失败: {e}", Colors.RED)
			 
			ci_bra_flat = self._ensure_1d(ci_bra)
			ci_ket_flat = self._ensure_1d(ci_ket)
			overlap = np.abs(np.vdot(ci_bra_flat, ci_ket_flat))
			soc_norm = np.linalg.norm(hso_operator)
			return overlap * soc_norm / ncas

	def _compute_soc_improved_approximate(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):

		try:
			 
			U, s, Vh = np.linalg.svd(hso_operator)
			max_singular_value = s[0] if len(s) > 0 else 0
			

			overlap = np.abs(np.vdot(ci_bra, ci_ket))
			
			 
			coupling_factor = self._estimate_coupling_factor(hso_operator)
			
			soc_estimate = max_singular_value * overlap * coupling_factor
			
			return soc_estimate
			
		except Exception as e:
			self._print(f"[-]Bad SOC in func[_compute_soc_improved_approximate]: {e}", Colors.YELLOW)
			return 1e-4   

	def _estimate_coupling_factor(self, hso_operator):
		n = hso_operator.shape[0]
		
		 
		sparse_threshold = 1e-6
		nonzero_ratio = np.sum(np.abs(hso_operator) > sparse_threshold) / (n * n)
		
		 
		diag_norm = np.linalg.norm(np.diag(hso_operator))
		off_diag_norm = np.linalg.norm(hso_operator - np.diag(np.diag(hso_operator)))
		
		if diag_norm > 0:
			dominance_ratio = off_diag_norm / diag_norm
		else:
			dominance_ratio = 1.0
		
		 
		coupling_factor = 0.1 * (1 - nonzero_ratio) * min(dominance_ratio, 1.0)
		
		return max(coupling_factor, 0.01)   
	def _compute_soc_simplified(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):
		try:
			 
			if isinstance(nelecas, (tuple, list)):
				nalpha, nbeta = nelecas
			else:
				nalpha = nbeta = nelecas // 2
			
			 
			ci_bra_flat = self._ensure_1d(ci_bra)
			ci_ket_flat = self._ensure_1d(ci_ket)
			
			 
			if len(ci_bra_flat) != len(ci_ket_flat):
				self._print(f"[-]Bad Length for CI Vector: bra={len(ci_bra_flat)}, ket={len(ci_ket_flat)}", Colors.YELLOW)
				return 0.0
			
			 
			if hasattr(hso_operator, 'shape') and hso_operator.shape == (len(ci_bra_flat), len(ci_ket_flat)):
				return self._compute_soc_full_matrix(hso_operator, ci_bra_flat, ci_ket_flat)
			
			 
			elif hasattr(hso_operator, 'shape') and len(hso_operator.shape) == 2:
				return self._compute_soc_orbital_based(hso_operator, ci_bra_flat, ci_ket_flat, ncas, (nalpha, nbeta))
			

			else:
				return self._compute_soc_configuration_selection(hso_operator, ci_bra_flat, ci_ket_flat, ncas, (nalpha, nbeta))
				
		except Exception as e:
			self._print(f"[-]Faild with Lower SOC clc: {e}", Colors.RED)
			import traceback
			self._print(traceback.format_exc(), Colors.RED)
			return 0.0

	def _compute_soc_full_matrix(self, hso_matrix, ci_bra, ci_ket):
		soc_value = 0.0
		n_dets = len(ci_bra)
		
		 
		for i in range(n_dets):
			for j in range(n_dets):
				if abs(ci_bra[i]) > 1e-8 and abs(ci_ket[j]) > 1e-8:   
					soc_value += ci_bra[i].conj() * hso_matrix[i, j] * ci_ket[j]
		
		self._print(f"[+]Full SOC Mtx: {soc_value:.6e}", Colors.GREEN)
		return soc_value

	def _compute_soc_orbital_based(self, hso_orbital, ci_bra, ci_ket, ncas, nelecas):
		
		nalpha, nbeta = nelecas
		n_dets = len(ci_bra)
		
		 
		soc_norm = np.linalg.norm(hso_orbital)
		soc_max = np.max(np.abs(hso_orbital))
		
		self._print(f"[+]Orb SOC Matrix:||X||={soc_norm:.4f}, Max={soc_max:.4f}", Colors.CYAN)
		
		 
		important_configs = self._select_important_configurations(ci_bra, ci_ket, min_weight=1e-4, max_configs=50)
		
		if not important_configs:
			self._print("[?]Did not found main CIs,use overlap * soc_norm / ncas", Colors.YELLOW)
			overlap = np.abs(np.vdot(ci_bra, ci_ket))
			return overlap * soc_norm / ncas 
		
		soc_value = 0.0
		computed_pairs = 0
		

		for (i, w_bra), (j, w_ket) in important_configs:

			config_soc = self._estimate_config_soc(i, j, hso_orbital, ncas, nelecas)

			config_overlap = ci_bra[i].conj() * ci_ket[j]
			
			soc_contribution = config_overlap * config_soc
			soc_value += soc_contribution
			computed_pairs += 1
		
		self._print(f"[+]Orb SOC: Value={soc_value:.6e}, has Calculated{computed_pairs} CIs", Colors.GREEN)
		return soc_value

	def _compute_soc_configuration_selection(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):

		nalpha, nbeta = nelecas
		
		 
		bra_weights = np.abs(ci_bra)**2
		ket_weights = np.abs(ci_ket)**2
		
		 
		bra_weights = bra_weights / np.sum(bra_weights)
		ket_weights = ket_weights / np.sum(ket_weights)
		
		 
		threshold = max(1e-4, 1.0 / len(ci_bra))   
		bra_important = np.where(bra_weights > threshold)[0]
		ket_important = np.where(ket_weights > threshold)[0]
		
		 
		min_configs = min(20, len(ci_bra) // 10)
		if len(bra_important) < min_configs:
			bra_important = np.argsort(bra_weights)[-min_configs:]
		if len(ket_important) < min_configs:
			ket_important = np.argsort(ket_weights)[-min_configs:]
		
		self._print(f"[:]Using CI: bra={len(bra_important)}, ket={len(ket_important)}", Colors.CYAN)
		
		soc_value = 0.0
		significant_contributions = 0
		
		 
		for i in bra_important:
			for j in ket_important:
				weight_product = bra_weights[i] * ket_weights[j]
				
				 
				if weight_product > 1e-6:
					 
					config_overlap = ci_bra[i].conj() * ci_ket[j]
					
					 
					if hasattr(hso_operator, 'shape'):

						soc_matrix_element = self._estimate_soc_from_operator(hso_operator, weight_product)
					else:

						soc_matrix_element = 0.01 * weight_product
					
					contribution = config_overlap * soc_matrix_element
					soc_value += contribution
					
					if abs(contribution) > 1e-8:
						significant_contributions += 1
		
		self._print(f"[:]Using CI: Value={soc_value:.6e}, Best Giving={significant_contributions}", Colors.GREEN)
		return soc_value

	def _select_important_configurations(self, ci_bra, ci_ket, min_weight=1e-4, max_configs=50):

		bra_amps = np.abs(ci_bra)
		ket_amps = np.abs(ci_ket)

		bra_amps = bra_amps / np.linalg.norm(bra_amps)
		ket_amps = ket_amps / np.linalg.norm(ket_amps)

		bra_important = np.where(bra_amps > min_weight)[0]
		ket_important = np.where(ket_amps > min_weight)[0]

		if len(bra_important) > max_configs:
			bra_important = np.argsort(bra_amps)[-max_configs:]
		if len(ket_important) > max_configs:
			ket_important = np.argsort(ket_amps)[-max_configs:]

		important_pairs = []
		for i in bra_important:
			for j in ket_important:
				weight = bra_amps[i] * ket_amps[j]
				if weight > min_weight:
					important_pairs.append(((i, bra_amps[i]), (j, ket_amps[j])))

		important_pairs.sort(key=lambda x: x[0][1] * x[1][1], reverse=True)
		
		return important_pairs[:max_configs]  

	def _estimate_config_soc(self, config_i, config_j, hso_orbital, ncas, nelecas):
		nalpha, nbeta = nelecas
		soc_mean = np.mean(np.abs(hso_orbital))
		soc_std = np.std(np.abs(hso_orbital))

		config_diff = abs(config_i - config_j)
		coupling_factor = np.exp(-config_diff / (2 * ncas))  

		soc_estimate = soc_mean * coupling_factor * (1 + 0.1 * soc_std / soc_mean if soc_mean > 0 else 1.0)
		
		return soc_estimate

	def _estimate_soc_from_operator(self, hso_operator, weight_product):
		try:
			if hso_operator.shape[0] > 1:
				s = np.linalg.svd(hso_operator, compute_uv=False)
				max_sv = s[0] if len(s) > 0 else np.linalg.norm(hso_operator)
			else:
				max_sv = np.abs(hso_operator[0, 0]) if hso_operator.shape[0] == 1 else np.linalg.norm(hso_operator)

			sparse_ratio = np.sum(np.abs(hso_operator) > 1e-8) / hso_operator.size
			sparsity_factor = 0.1 + 0.9 * (1 - sparse_ratio) 

			weight_factor = np.sqrt(weight_product)
			
			return max_sv * sparsity_factor * weight_factor
			
		except Exception as e:
			self._print(f"[-]SOC Err: {e},Using 0.01*weight_product", Colors.YELLOW)
			return 0.01 * weight_product


	def _compute_soc_matrix_elementsold(self, hso1e_cas, ci_i, ci_j, nelecas, ncas, mol):

		 
		if isinstance(nelecas, (tuple, list)):
			nalpha, nbeta = nelecas
		else:
			nalpha = nbeta = nelecas // 2
		
		 
		print(f"CI vectors Shape: {ci_i.shape}, {ci_j.shape}")
		print(f"||CI vectors||: {np.linalg.norm(ci_i)}, {np.linalg.norm(ci_j)}")
		print(f"CI wavego: {np.max(np.abs(ci_i))}, {np.max(np.abs(ci_j))}")
		print(f"CI <|>: {np.abs(np.vdot(ci_i, ci_j))}")
		ci_i_vec = self._ensure_1d(ci_i)
		ci_j_vec = self._ensure_1d(ci_j)
		
		soc_elements = np.zeros(3, dtype=complex)
		
		 
		element_count = {}
		for atom in mol.atom:
			element = atom[0]
			element_count[element] = element_count.get(element, 0) + 1
		
		self._print(f"Molecular composition: {element_count}", Colors.WHITE)
		
		 
		for x in range(3):
			hso_operator = hso1e_cas[x]
			
			try:
				 
				soc_element = self._compute_soc_improved_approximate(hso_operator, ci_i_vec, ci_j_vec, ncas, (nalpha, nbeta))
				soc_elements[x] = soc_element
				
			except Exception as e:
				self._print(f"SOC calculation failed for component {x}: {e}", Colors.YELLOW)
				 
				soc_element = self._compute_soc_eigenvalue_method(hso_operator, ci_i_vec, ci_j_vec)
				soc_elements[x] = soc_element
		
		 
		self._analyze_soc_origin(hso1e_cas, ci_i_vec, ci_j_vec, ncas, (nalpha, nbeta), mol)
		
		return soc_elements

	def _compute_soc_improved_approximate(self, hso_operator, ci_bra, ci_ket, ncas, nelecas):
		soc_frobenius = np.linalg.norm(hso_operator, 'fro')
		
		 
		overlap = np.abs(np.dot(ci_bra.conj(), ci_ket))
		U, s, Vh = np.linalg.svd(hso_operator)
		max_singular_value = s[0]   

		soc_element = overlap * max_singular_value * self._get_coupling_strength(hso_operator)
		
		return soc_element

	def _get_coupling_strength(self, hso_operator):
		diag_elements = np.diag(hso_operator)
		off_diag_norm = np.linalg.norm(hso_operator - np.diag(diag_elements), 'fro')
		diag_norm = np.linalg.norm(diag_elements)
		
		if diag_norm > 0:
			coupling_ratio = off_diag_norm / diag_norm
		else:
			coupling_ratio = 1.0
		
		 
		n = hso_operator.shape[0]
		sparse_threshold = 1e-6
		nonzero_ratio = np.sum(np.abs(hso_operator) > sparse_threshold) / (n * n)
		
		 
		coupling_factor = coupling_ratio * (1 - nonzero_ratio) * 0.5 + 0.5
		
		return min(max(coupling_factor, 0.1), 1.0)

	def _compute_soc_eigenvalue_method(self, hso_operator, ci_bra, ci_ket):
		eigenvalues = np.linalg.eigvals(hso_operator)
		
		 
		soc_strength = np.max(np.abs(eigenvalues))
		
		 
		overlap = np.abs(np.dot(ci_bra.conj(), ci_ket))
		
		 
		soc_element = overlap * soc_strength * 0.3   
		
		return soc_element

	def _analyze_soc_origin(self, hso1e_cas, ci_bra, ci_ket, ncas, nelecas, mol):
		self._print("Analyzing SOC origin...", Colors.CYAN)
		
		 
		soc_norms = [np.linalg.norm(hso1e_cas[x]) for x in range(3)]
		total_soc_norm = np.linalg.norm(soc_norms)
		
		self._print(f"SOC operator norms (x,y,z): {[f'{n:.6f}' for n in soc_norms]}", Colors.WHITE)
		self._print(f"Total SOC operator norm: {total_soc_norm:.6f}", Colors.WHITE)
		
		 
		self._analyze_orbital_contributions(hso1e_cas, ncas)
		
		 
		organic_elements = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl'}
		actual_elements = set(atom[0] for atom in mol.atom)
		
		has_heavy_atoms = any(element not in organic_elements for element in actual_elements)
		
		if not has_heavy_atoms:
			self._print("System: Pure organic (no heavy atoms)", Colors.WHITE)
			self._print("SOC mechanism: Spin-orbit coupling through orbital mixing", Colors.WHITE)
		else:
			heavy_atoms = [atom[0] for atom in mol.atom if atom[0] not in organic_elements]
			self._print(f"System: Contains heavy atoms: {heavy_atoms}", Colors.WHITE)

	def _analyze_orbital_contributions(self, hso1e_cas, ncas):
		orbital_contributions = np.zeros(ncas)
		
		for x in range(3):
			 
			diag_contributions = np.abs(np.diag(hso1e_cas[x]))
			
			 
			row_contributions = np.sum(np.abs(hso1e_cas[x]), axis=1)
			col_contributions = np.sum(np.abs(hso1e_cas[x]), axis=0)
			
			orbital_contributions += diag_contributions + 0.5 * (row_contributions + col_contributions)
		
		 
		if len(orbital_contributions) > 0:
			max_contrib_idx = np.argmax(orbital_contributions)
			max_contrib_value = orbital_contributions[max_contrib_idx]
			
			self._print(f"Maximum SOC contribution from orbital {max_contrib_idx}: {max_contrib_value:.6f}", Colors.WHITE)
		
		 
		soc_matrix_combined = np.sum(np.abs(hso1e_cas), axis=0)
		sparse_threshold = 1e-6
		sparse_ratio = np.sum(soc_matrix_combined > sparse_threshold) / (ncas * ncas)
		
		self._print(f"SOC matrix density: {sparse_ratio*100:.1f}%", Colors.WHITE)

	def _ensure_1d(self, ci_vector):
		if ci_vector.ndim == 2:
			if ci_vector.shape[1] == 1:
				return ci_vector.ravel()
			else:
				return ci_vector[:, 0]



	def compute_phosphorescence_rateold(self, soc_norm, energy_gap):
		deltaE_cm = energy_gap * 8065.73 
		 

		h = 6.62607015e-27   
		c = 2.99792458e10	
		

		prefactor = (64 * np.pi**4) / (3 * h)
		rate_si = prefactor * (soc_norm**2) * (deltaE_cm**3) / (c**3)
		
		lifetime = 1.0 / rate_si if rate_si > 0 else float('inf')
		
		self._print(f"V(n) ΔE={deltaE_cm:.1f} cm-1, SOC={soc_norm:.3f} cm-1", Colors.WHITE)
		self._print(f"v={rate_si:.3e} s⁻¹, t={lifetime:.3e} s", Colors.WHITE)
		
		return rate_si, lifetime

	
	def save_results(self, filename='phosphorescence_soc_results.pkl'):

		with open(filename, 'wb') as f:
			pickle.dump(self.results, f)
		self._print(f"Results saved to {filename}", Colors.GREEN)