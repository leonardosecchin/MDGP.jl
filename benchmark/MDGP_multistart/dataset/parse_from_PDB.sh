#!/bin/bash

rm -rdf data
rm -rdf output

probs=(
"1E0Q"
"1HO7"
"1LFC"
"1MMC"
"6HKA"
"1SPF"
"1C56"
"2LKS"
"7EAU"
"6QBK"
"2KNX"
"7PQW"
"2EGE"
"2YRT"
"2KW9"
"1KCY"
"2N2N"
"2N9D"
"2EBT"
"1J0F"
"8VRC"
"2VB5"
"2J4M"
"1DX0"
"1A66"
"1BC9"
"1JCU"
"1AP8"
"1EZA"
"1AH2"
)

for p in "${probs[@]}"; do
    echo "Parsing $p"
    python3 pdb_parser.py --data_dir data --output_dir output --pdb_id $p --model 1 --chain A --ddgp_hc_order 1 --cut 5.0 --interval_width 2.0 --local_interval_width 1.0 --angular_width 50 1
done
