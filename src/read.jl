# compute torsions angles from X
function omega(l, P, X)
    l1, l2, l3 = P[l,1:3]

    a = X[1:3,l2] - X[1:3,l3]
    b = X[1:3,l1] - X[1:3,l2]
    c = X[1:3,l ] - X[1:3,l1]

    return 180 * atan(
        norm(b) * dot(a, cross(b, c)),
        dot(cross(a, b), cross(b, c))
    ) / pi
end

"""
    X, Dij, D, P, residues, atoms, torsions = MDGP_read(Dfile, Pfile; Xfile = "", recompute_signs = true)

Read an MDGP instance. Paths for distance (`Dfile`) and predecessor
(`Pfile`) files must be provided. Optionally, a reference solution file
(`Xfile`) can be specified. If given, signs in `Pfile` are recomputed when
`recompute_signs = true`.

## Output
- `X`: reference solution `3 x |atoms|` matrix
- `Dij`: `|distances| × 2` matrix with indices `(i, j)` for each distance `d_ij`
- `D`: `|distances| × 2` matrix with lower and upper bounds for each distance
- `P`: predecessor matrix
- `residues`: residue index of each atom
- `atoms`: atom types
- `torsions`: `|atoms| × 2` matrix with reference torsion angles and their shifts

## Example

Download files `D_1TOS.dat` and `P_1TOS.dat` from

`https://github.com/leonardosecchin/MDGP/tree/main/test`.

From the same folder, run

`_, Dij, D, P, residues, atoms, torsions = mdgp_read("D_1TOS.dat", "P_1TOS.dat")`
"""
function mdgp_read(
    Dfile::String, Pfile::String; Xfile::String = "", recompute_signs = true
)
    Df = isfile(Dfile)
    Pf = isfile(Pfile)
    Xf = isfile(Xfile)

    check_param(Df, "Invalid distance file")
    check_param(Pf, "Invalid predecessor file")

    if !isempty(Xfile) && !Xf
        @warn "Solution file not found, ignoring..."
    end

    Dij = []
    D = []
    X = []
    P = []
    res = []
    atoms = []
    resnames = []
    angles = []

    Ddata = readdlm(Dfile)
    Pdata = readdlm(Pfile)

    P = Int64.(Pdata[:,2:5])
    Dij = Int64.([Ddata[:,2] Ddata[:,1]])   # the second indice is read first

    # reindex Dij and P
    min_i = minimum(Dij)
    Dij .-= min_i - 1
    P[:,1:4] .-= min_i - 1

    # number of atoms
    n = maximum(Dij)

    check(n == size(P,1), "Invalid predecessor list mismatch or non-consecutive atom indices")

    D = Float64.([Ddata[:,5] Ddata[:,6]])
    res = Int64.([Ddata[:,4] Ddata[:,3]])
    atoms = [Ddata[:,8] Ddata[:,7]]
    resnames = [Ddata[:,10] Ddata[:,9]]

    # read the reference solution
    if Xf
        Xdata = readdlm(Xfile)
        X = Matrix(Float64.(Xdata[:,1:3]'))
    end

    # recompute signs if X is available
    if Xf && recompute_signs
        @inbounds for i in 4:n
            if P[i,4] != 0
                P[i,4] = Int64(sign(omega(i,P,X)))
            end
        end
    end
    torsions = abs.(Float64.(Pdata[:,6:7]))

    # read distances, dividing them by type
    idxDpred = Int64[]      # diatances between predecessors
    idxDnonpred = Int64[]   # other with lower and upper bounds
    idxDvdw = Int64[]       # Van der Waals (only lower bound is present)
    @inbounds for k in 1:size(D,1)
        i, j = Dij[k,1], Dij[k,2]
        if (j in P[i,1:3]) || (i in P[j,1:3])
            push!(idxDpred, k)
        elseif D[k,2] < 900.0
            push!(idxDnonpred, k)
        else
            push!(idxDvdw, k)
        end
    end

    # sort
    TMP = sortslices([Dij D res atoms resnames][idxDpred,:], dims=1, by=row -> (row[2],row[1]))
    TMP = [TMP; sortslices([Dij D res atoms resnames][idxDnonpred,:], dims=1, by=row -> (row[2],row[1]))]
    TMP = [TMP; sortslices([Dij D res atoms resnames][idxDvdw,:], dims=1, by=row -> (row[2],row[1]))]

    Dij = Int64.(TMP[:,1:2])
    D = Float64.(TMP[:,3:4])
    res = Int64.(TMP[:,5:6])
    atoms = TMP[:,7:8]
    resnames = TMP[:,9:10]

    # make explicit infinite upper bounds
    D[ D .>= 900.0 ] .= Inf

    # map between atoms indices to their types and residues
    atoms_map = fill("?", n)
    res_map = zeros(Int64, n)
    @inbounds @views for k in 1:size(Dij,1)
        atoms_map[Dij[k,1]] = strip(atoms[k,1])
        atoms_map[Dij[k,2]] = strip(atoms[k,2])
        res_map[Dij[k,1]] = res[k,1]
        res_map[Dij[k,2]] = res[k,2]
    end

    return X, Dij, D, P, res_map, atoms_map, torsions
end
