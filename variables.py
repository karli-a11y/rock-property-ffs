"""variables.py — the 19 standardised variables of the harmonised master table.

Central definition of the variable universe used by every analysis script:

* ``NUMERIC_FEATURES``    — column names of the 19 standardised variables
                            (17 of them also qualify as prediction targets,
                            see config/targets.yaml).
* ``FEATURE_LABELS``      — display labels (LaTeX symbols and units) used in
                            all figures and tables.
* ``HARMONIZATION_RISK``  — harmonisation-risk class per variable (low /
                            moderate / high), reflecting alias ambiguity,
                            unit-conversion demands, and protocol
                            heterogeneity across the source databases.

The values mirror Table 1 of the accompanying paper.
"""

NUMERIC_FEATURES = [
    "porosity_pct",
    "bulk_density_gcm3", "grain_density_gcm3",
    "permeability_m2",
    "tc_dry_WmK",
    "td_dry_1e6m2s",
    "cp_JkgK",
    "vp_dry_ms", "vs_dry_ms",
    "ucs_MPa", "E_static_GPa", "E_dyn_GPa",
    "poisson_ratio", "tensile_MPa",
    "friction_angle_deg", "cohesion_MPa",
    "mag_susc_1e3si", "resistivity_dry_Ohm",
    "depth_m",
]

FEATURE_LABELS = {
    "porosity_pct":            r"Porosity $\phi$ (%)",
    "bulk_density_gcm3":       r"Bulk density $\rho_b$ (g/cm³)",
    "grain_density_gcm3":      r"Grain density $\rho_g$ (g/cm³)",
    "permeability_m2":         r"Permeability $k$ (m²)",
    "tc_dry_WmK":              r"Thermal conductivity $\lambda$ (W/m·K)",
    "td_dry_1e6m2s":           r"Thermal diffusivity $\alpha$ ($\times 10^{-6}$ m²/s)",
    "cp_JkgK":                 r"Specific heat $c_p$ (J/kg·K)",
    "vp_dry_ms":               r"P-wave velocity $V_P$ (m/s)",
    "vs_dry_ms":               r"S-wave velocity $V_S$ (m/s)",
    "ucs_MPa":                 r"UCS (MPa)",
    "E_static_GPa":            r"Static Young's modulus $E_s$ (GPa)",
    "E_dyn_GPa":               r"Dynamic Young's modulus $E_d$ (GPa)",
    "poisson_ratio":           r"Poisson ratio $\nu$",
    "tensile_MPa":             r"Tensile strength $\sigma_t$ (MPa)",
    "friction_angle_deg":      r"Friction angle $\varphi$ (°)",
    "cohesion_MPa":            r"Cohesion $c$ (MPa)",
    "mag_susc_1e3si":          r"Magnetic susceptibility $\chi$ ($\times 10^{-3}$ SI)",
    "resistivity_dry_Ohm":     r"Electrical resistivity $\rho_e$ (Ω·m)",
    "depth_m":                 r"Depth (m)",
}

HARMONIZATION_RISK = {
    "porosity_pct": "moderate",
    "bulk_density_gcm3": "low", "grain_density_gcm3": "low",
    "permeability_m2": "high",
    "tc_dry_WmK": "moderate", "td_dry_1e6m2s": "moderate",
    "cp_JkgK": "low",
    "vp_dry_ms": "moderate", "vs_dry_ms": "moderate",
    "ucs_MPa": "high", "E_static_GPa": "high", "E_dyn_GPa": "moderate",
    "poisson_ratio": "high", "tensile_MPa": "high",
    "friction_angle_deg": "high", "cohesion_MPa": "high",
    "mag_susc_1e3si": "low", "resistivity_dry_Ohm": "moderate",
    "depth_m": "low",
}
