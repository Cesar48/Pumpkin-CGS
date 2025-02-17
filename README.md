# Core-Guided Pumpkin

A fork of [Pumpkin](https://github.com/ConSol-Lab/pumpkin) implementing core-guided search. All original linear search features have been retained; see the [original README](https://github.com/ConSol-Lab/Pumpkin/blob/main/README.md) for an overview of these features.

**The `--core-guided` flag is required to use core-guided search. When this flag is absent, the (original) linear search version of Pumpkin will be run.**

Since most information from the [original README](https://github.com/ConSol-Lab/Pumpkin/blob/main/README.md) is still relevant for this version of Pumpkin, we will not repeat it here. Only the parts specifically relevant for the core-guided implementation will be discussed.

## Core-guided approaches
Gange et al. (2020) defines two reformulation approaches and two weight handling approaches for core-guided search in CP. Of these, slice-based reformulations and weight splitting most closely resemble the approach usually taken in MaxSAT - where core-guided search originates. As such, the combination of these two approaches is used as the default. The other approaches, variable-based reformulations and coefficient elimination, can be enabled by using the flags `--reform-var` and `--coef-elim`, respectively. Each of the four combinations is compatible with any feature.

## Core-guided features
As part of the research developing this implementation, four additional features for core-guided search were implemented. These are the following:
 - Weight-aware Core Extraction (WCE), enabled through flag `--wce` (used previously in Gange et al. 2020)
 - Stratification, enabled through flag `--strat` (used previously in Gange et al. 2020)
 - Hardening, enabled through flag `--harden` (used previously in Gange et al. 2020)
 - Partitioning, enabled through flag `--partition` (newly implemented based on Open-WBO, Neves et al. 2015)

These features are compatible with one another, though combining partitioning with stratification is not recommended: as both consider only a (small) part of the objective function at first, this causes very relaxed versions of the problem to be considered at first. This can be detrimental to the search process. Additionally, WCE and stratification both progress as soon as an intermediate solution is discovered, even if progressing just one of the techniques would allow new cores to be discovered. This may have unintended side effects, though we expect there to still be significant benefits in most cases.



Partitioning makes use of the [louvain-rs](https://github.com/graphext/louvain-rs) library to calculate partitions. 
