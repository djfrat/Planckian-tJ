# Planckian-tJ
Data for the manuscript "Strange metal behavior from coupling to spin fluctuations"

## Experimental two-current / Spin(2,3) response extension

`spin23_two_current.py` adds the model-independent machinery for promoting a
scalar current spectrum to a symmetric two-current response matrix.  For a
physical thermoelectric calculation the operators should be the particle
current `J_N` and an energy current `J_E` derived from a declared local-energy
continuity equation.  The module provides:

- a direct two-current Lehmann evaluator for exact-diagonalization checks;
- reconstruction of the symmetric cross spectrum from three scalar FTLM runs
  for `J_N`, `J_E`, and `J_N + J_E`;
- the grand-canonical change of basis `J_Q = J_E - mu J_N` as a matrix
  congruence;
- detailed-balance, positivity, Lorentz-determinant, and open-circuit Schur
  complement checks; and
- an explicitly labelled `{H,J_N}/2` diagnostic proxy.  This proxy is not the
  interacting t-J energy current and must not be used as a thermal-transport
  prediction.

Run the dependency-light checks with:

```text
python -m unittest -v test_spin23_two_current.py
```

For the existing FTLM implementation, call its scalar auto-spectrum path three
times and pass the resulting arrays to `assemble_by_polarization`.  This avoids
a new cross-Krylov estimator.  Positivity should be checked after identical
flux, random-vector, broadening, and ensemble averaging have been applied to
all three spectra.

The original program now accepts two optional arguments after `Kinb`:

```text
charge|energy_proxy|sum_proxy   energy_scale
```

The default is `charge 1.0` and retains the original filenames and behavior.
The proxy modes use `{H,J_N}/(2 energy_scale)` and add an operator label to
their output filenames.  Use the same positive `energy_scale` in all three
runs.  The regular optical spectrum is the diagnostic output in proxy modes;
the program's Drude and kinetic sum-rule columns retain their charge-current
formula and are not physical energy-current quantities.  Afterward assemble
matching output spectra with:

```text
python assemble_spin23_ftlm.py CHARGE.dat ENERGY_PROXY.dat SUM_PROXY.dat MATRIX.dat
```

`MATRIX.dat` contains the three response entries, determinant, smallest
eigenvalue, and open-circuit Schur complement.  A negative determinant or
smallest eigenvalue beyond sampling error signals inconsistent averaging,
insufficient convergence, or failure of the proposed positive carrier.

The resulting positive matrix belongs kinematically to `Sym_2(R)_+`.  Its
Jordan conformal completion is `Sp(4,R) ~= Spin_0(2,3)`; this alone does not
establish that the full group is a dynamical symmetry of the t-J model.
