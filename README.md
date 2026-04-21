# MDGP

This is an implementation of the multistart strategy to solve the Molecular
Distance Geometry Problem (MDGP) with interval data, as described in

[Secchin, da Rocha, da Rosa, Liberti, Lavor. A hybrid combinatorial-continuous strategy for solving molecular distance geometry problems. 2025](https://arxiv.org/abs/2510.19970)

## Installation

`]add https://github.com/leonardosecchin/MDGP.git`

## Usage

The basic usage is

`mdgp_multistart(Dij, D, P, atoms, torsions)`

- `Dij` is the `nd x 2` matrix of the indices `i,j` of the distances, where `nd`
is the number of distances.
- `D` is the `nd x 2` matrix of corresponding distances intervals
`[lower d_ij, upper d_ij]`.
- `P` is a `nv x 4`, where `nv` is the number of
atoms. `P[i,1:3]` are the indices of predecessors of atom `i` in descending
order, while `P[i,4]` is `1`, `-1` or `0`, indicating the
"side" that atom `i` is located with respect to the plane of its predecessors
(quirality); when `P[i,4] = 0`, both sides are accepted.
- `atoms` is the `nv` vector containing the name of each atom (for example,
"H1", "N", "CA", "HA", "C" and so on).
- `torsions` is the `nv x 2` matrix whose i-th row `[w, delta]` represents the
torsion angle interval in the format `[w-delta, w+delta]` for atom `i`, in
degrees. `w` must be non-negative (it sign is given by `P`).

For more details, run `?mdgp_multistart`.

### Changing parameters

You can modify the algorithm parameters described in the reference paper. For
futher details, run `?mdgp_multistart`.

## Reading instances

The instances used in the reference paper are located in the
`benchmark/MDGP_multistart/dataset` folder. Each instance consists of three
files, prefixed with `I_`, `T_` and `X_`. The first is the distance file
(`Dfile`), the second contains the predecessors (`Pfile`) of each atom, and the
last provides the reference solution extracted from the PDB (`Xfile`). These
files can be loaded into Julia using

`X, Dij, D, P, residues, atoms, torsions = mdgp_read("path to I_ file", "path to T_ file"; Xfile = "path to X_ file")`

For more details, run `?mdgp_read`.

You can also generate instances from the [PDB](https://www.rcsb.org/) using the
Python parser developed by Wagner da Rocha. This parser produces the
distance and predecessor matrices, as well as the reference solution file. It is
included in the `benchmark/MDGP_multistart` directory. For additional details and
updates, please visit [https://github.com/wdarocha/pdb-parser].

Scripts for reproducing the tests reported in the reference are available in the
`benchmark/MDGP_multistart` folder. Note that the packages required by these
scripts are not necessarily included among the dependencies of the MDGP package,
and therefore must be installed manually.

## Funding

This research was partially supported by the São Paulo Research Foundation
(FAPESP) (grant 2024/12967-8) and the National Council for Scientific and
Technological Development (CNPq) (grant 302520/2025-2), Brazil.

## How to cite

If you use this code in your publications, please cite us, see the
`CITATIONS.bib` in the repository.