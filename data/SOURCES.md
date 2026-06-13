# Source databases

The master table merges 30 source databases. The original raw files are
**not** redistributed in this repository. The table below lists each source
with its row count in the deduplicated master table and, where the source is a
public dataset, its DOI. Row-level provenance is preserved in the master table
itself through the `source_db`, `parent_db`, `primary_reference`, and
`subsource` columns, so every record can be traced back to its primary
reference.

| Source (`source_db`) | Rows | Scope | Public dataset DOI / reference |
|---|---:|---|---|
| GlobalDB | 7,468 | CSIRO Global Petrophysical Database v5 | [10.25919/KFVT-2197](https://doi.org/10.25919/KFVT-2197) |
| P3 | 6,358 | P³ PetroPhysical Property Database v1.0 | [10.5880/GFZ.4.8.2019.P3](https://doi.org/10.5880/GFZ.4.8.2019.P3) |
| USGS_WesternUS_Phillips2021 | 3,172 | Petrophysical database of rocks from the western USA and Alaska | [10.5066/P9C7HDFR](https://doi.org/10.5066/P9C7HDFR) |
| AMPEDEK | 2,715 | Atlas of geochemical, petrophysical and mechanical rock properties of the German crystalline basement | DOI forthcoming (TU Darmstadt) |
| MidGerman | 2,226 | Mid-German Crystalline Rise petrophysical database | [10.25534/tudatalib-406](https://doi.org/10.25534/tudatalib-406) |
| Weydt2020 | 1,999 | Los Humeros and Acoculco geothermal fields (Mexico) | [10.25534/tudatalib-201.10](https://doi.org/10.25534/tudatalib-201.10) |
| USGS_MojaveREE_2020 | 1,033 | USGS Mojave REE deposits petrophysics | USGS data release (see `primary_reference`) |
| Valgarður | 701 | Valgarður database of Icelandic rocks | [10.5281/zenodo.6980231](https://doi.org/10.5281/zenodo.6980231) |
| Mielke2017 | 665 | Thermal conductivity and P-wave velocity dataset | [10.1594/PANGAEA.874146](https://doi.org/10.1594/PANGAEA.874146) |
| HeapViolay2021 | 644 | Compilation of volcanic-rock mechanical properties | see `primary_reference` column |
| USGS_GreatBasinPlutons | 598 | USGS Great Basin plutons petrophysics | USGS data release (see `primary_reference`) |
| Cornwall_Turan2024 | 266 | Cornubian Batholith petrophysics | see `primary_reference` column |
| ESReviews2024_VolcGeotherm | 149 | Volcanic/geothermal review compilation | see `primary_reference` column |
| GenevaBasin | 148 | Geneva Basin CERN geomechanical dataset | [10.5281/zenodo.4725585](https://doi.org/10.5281/zenodo.4725585) |
| AbuDhabiEvaporite2024 | 143 | Abu Dhabi evaporite sequence | see `primary_reference` column |
| Wyering2014_TVZ | 131 | Taupō Volcanic Zone geothermal rock properties | see `primary_reference` column |
| ChainePuysHeap2024 | 45 | Chaîne des Puys volcanic rocks | see `primary_reference` column |
| CentralApennines | 38 | Central Apennines limestones | [10.4121/21533988.v1](https://doi.org/10.4121/21533988.v1) |
| SchaeferPacaya2015 | 22 | Pacaya volcano basalts | see `primary_reference` column |
| SKB_Forsmark | 13 | SKB Forsmark site characterisation | SKB report series |
| UnitedDowns_Cornwall | 13 | United Downs deep geothermal project | see `primary_reference` column |
| ApuaniStromboli2005 | 12 | Stromboli volcanics | see `primary_reference` column |
| FrolovaKurilKamchatka | 12 | Kuril–Kamchatka volcanics | see `primary_reference` column |
| AydanWeldedTuff | 11 | Welded tuff mechanical properties | see `primary_reference` column |
| SKB_Laxemar | 8 | SKB Laxemar site characterisation | SKB report series |
| KlodawaSalt2024 | 6 | Kłodawa rock salt | see `primary_reference` column |
| WuMudstone2025 | 6 | Mudstone dataset | see `primary_reference` column |
| UnzenHeap2021 | 5 | Unzen volcano dacites | see `primary_reference` column |
| KimSciRep2026 | 3 | Rock property dataset | see `primary_reference` column |
| PosivaOlkiluoto | 2 | Posiva Olkiluoto site characterisation | Posiva report series |
| NorwayNPD | (excluded) | Norwegian Petroleum Directorate porosity–permeability database, contained in the master table but excluded from all analyses | [10.5281/zenodo.4419060](https://doi.org/10.5281/zenodo.4419060) |

Row counts refer to the deduplicated master table after exclusion of
NorwayNPD (28,612 effective rows, 267 distinct primary references). The full
per-source × lithology-class matrix is given in Supplementary Table S1 of the
paper, and the associated journal publications of the public datasets are
cited in the paper's reference list.

Each original database remains under the licence of its publisher. Users who
redistribute subsets of the master table must comply with the per-source
licences linked above.
