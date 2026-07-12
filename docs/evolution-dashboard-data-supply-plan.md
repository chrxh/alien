# Evolution Dashboard — Data Supply Plan

Plan for connecting the Evolution Dashboard prototype (`source/Gui/EvolutionDashboardWindow.*`,
currently fed by `generateDummyData()`) to real simulation data. No implementation yet.

Analyzed state: `origin/features/EvolutionDashboard` (42c30c25f).

## Current architecture (as-is)

Statistics pipeline:

```
StatisticsKernels.cu (GPU, per update)      SimulationStatistics.cuh (device accumulators)
        │ writes StatisticsRawData { TimelineStatistics { timestep, accumulated }, histogram }
        ▼
SimulationCudaFacade::updateStatistics
        ├─► StatisticsService::addDataPoint ──► StatisticsConverterService::convert
        │        └─► StatisticsHistory (long-term, downsampled, persisted as .statistics.csv)
        └─► getStatisticsRawData() ──► GUI (StatisticsWindow::processBackground)
                 ├─► TimelineLiveStatistics (real-time window, wall-clock based)
                 ├─► TableLiveStatistics    (rates per second from accumulated counters)
                 └─► HistogramLiveStatistics
```

Relevant engine structures (feature branch):

- `Creature` (Entities.cuh): `generation`, `numCells`, `genome`, `lineageId`,
  `accumulatedMutations` (reset to 0 when a new lineage is formed in
  `MutationProcessor::updateAccumulatedMutationsAndLineageId`), temp field `creatureIndex`.
- `Genome` (Genome.cuh): `numGenes`, `genes`, `mutationRates`, temp field `genomeIndex`.
- `MutationRates`: 13 `nodeProbability` fields (neuron ×2, connection ×2,
  cellTypeProperties ×2, cellTypeMode, cellType, void, addNode, deleteNode, constructor ×2)
  and 8 `geneProbability` fields (geometry ×2, extendGene, trimGene, duplicateGene,
  deleteGene, copyNodeSection, moveNodeSection).
- Creatures and genomes are only reachable via `Cell::creature` — there is no global array.
  `DataAccessKernels.cu` already solves per-creature/per-genome deduplication with
  `atomicExch` sentinels on the temporary `creatureIndex` / `genomeIndex` members.

Existing code stays untouched; all statistics additions are additive.

## Dashboard columns → data sources

| Column | Definition | Source |
|---|---|---|
| Creatures | number of creatures | count deduplicated `Creature`s |
| ~~Avg size~~ | removed | drop from `Metrics[]`, `NumMetrics` 9 → 8 |
| Avg cells | average of `Creature::numCells` | sum + count per update |
| Avg nodes | average node count of all genomes | per genome: Σ `gene->numNodes` (cf. `MutationProcessor::getNumberOfNodes`) |
| Sum energy | total energy of all creature cells | Σ energy over `ObjectType_Cell` objects |
| Avg mut. rate | average of all `nodeProbability` + `geneProbability` values of all genomes (21 values each) | per-genome mean, summed over genomes |
| Avg age (gen) | average of `Creature::generation` | sum + count per update |
| Created /s | newly created creatures per second | accumulated counter, rate derived in GUI |
| Mutations /s | change of total accumulated mutations per second | accumulated quantity, rate derived in GUI |

Principle (as requested): `SimulationStatistics` collects only momentary sums/counts and
monotonically accumulated quantities. Rates ("/s") are derived GUI-side from consecutive
samples using wall-clock deltas (`DataPointCollection::systemClock` already exists).

## Phase 0 — `Creature::accumulatedMutationsInLineage` refactoring (separate PR)

Goal: `accumulatedMutations` becomes a never-reset total; the lineage-switch logic moves to a
new member.

- `Entities.cuh` (`Creature`), `TOs.cuh` (`CreatureTO`), `Desc.h` (`CreatureDesc`):
  add `float accumulatedMutationsInLineage`.
- `MutationProcessor::updateAccumulatedMutationsAndLineageId`: add the per-mutation delta to
  **both** members; the `newLineageThreshold` check and the reset apply only to
  `accumulatedMutationsInLineage`.
- Data transfer: `EntityFactory.cuh` (TO → `Creature`, ~line 619),
  `DataAccessKernels.cu` (`Creature` → TO, ~line 301), `DescConverterService.cpp`
  (TO ↔ Desc, creature section).
- Serialization (`SerializerService.cpp`): new `Id_Creature_AccumulatedMutationsInLineage`.
  Creature ids 0–7 and 9 are in use; id 8 looks like a historical gap — use a fresh id (10)
  to be safe. **Migration:** when loading files without the new field, set
  `accumulatedMutationsInLineage = accumulatedMutations` (old semantics); keep the loaded
  value in `accumulatedMutations` as a lower bound of the true total.
- GUI: `InspectionWindow.cpp` currently shows `accumulatedMutations`; switch to (or
  additionally show) `accumulatedMutationsInLineage`, since `newLineageThreshold` refers to it.
- Validation (`DescValidationService`), `DescTestDataFactory`, `GenomeDescHash` /
  creature-desc hashing if applicable, and tests (`AccumulatedMutationTests.cpp`: lineage
  switch resets the in-lineage value, total keeps growing).
- Note: `EntityFactory::cloneCreature` copies both members via `*newCreature = *creature`
  — offspring inherit the parent's totals, which is consistent with lineage semantics.

## Phase 1 — engine-side collection in `SimulationStatistics` (additive)

New members in `StatisticsRawData.h`, e.g. as a nested `EvolutionStatistics` struct inside
`TimestepStatistics` (reset every statistics update):

- `numCreatures`, `sumCreatureCells`, `sumCreatureGenerations`
- `numGenomes`, `sumGenomeNodes`, `sumMutationRates` (per-genome mean of the 21
  probability fields, summed over genomes)
- `sumCreatureEnergy` (energy of all `ObjectType_Cell` objects)
- `sumAccumulatedMutations` (Σ `Creature::accumulatedMutations`, never-reset variant from
  Phase 0)

New member in `AccumulatedStatistics` (monotonic, never reset):

- `numCreatedCreatures` — incremented where offspring creatures are created:
  `EntityFactory::cloneCreature` call site in `ConstructorProcessor.cuh` (~line 275).
  The enclosing function does not receive `statistics` yet; thread it through analogous to
  `incNumCreatedCells`.

Collection in `StatisticsKernels.cu`, new substeps using the proven dedup pattern from
`DataAccessKernels.cu`:

1. Substep A: iterate objects; for cells set `creature->creatureIndex` and
   `genome->genomeIndex` to a sentinel (`VALUE_NOT_SET_UINT64`).
2. Substep B: iterate cells; `atomicExch` on the sentinel — only the first thread per
   creature adds creature-level values (`numCells`, `generation`, `accumulatedMutations`),
   only the first per genome adds genome-level values (nodes, mutation rates).
   Cell energy is added per cell (no dedup needed).

Both fields are documented as temporary ("May be invalid"), and statistics kernels run
between timesteps, so this is safe. Cost is O(objects), same as the existing substep 2.

Caveats:

- Creatures under construction (`numCells == 0`) should probably be excluded (open question).
- `.cuh` struct changes require a clean rebuild (stale-kernel rule from CLAUDE.md).

## Phase 2 — transport to the GUI (global values: "All lineages" row, timelines, KPI cards)

- `DataPointCollection`: new fields — momentary values (`numCreatures`,
  `averageCreatureCells`, `averageGenomeNodes`, `sumCreatureEnergy`, `averageMutationRate`,
  `averageGeneration`) and raw accumulated values (`accumNumCreatedCreatures`,
  `accumMutations`). Rates are *not* stored.
- `StatisticsConverterService::convert`: averages from sums/counts; accumulated values passed
  through unchanged.
- Rate derivation in the dashboard: `(accum[t] − accum[t−1]) / Δ systemClock` for
  "Created /s" and "Mutations /s"; clamp negative deltas to 0 (they occur after
  `resetAccumulatedStatistics`).
- **Known artifact** of the sum-based "Mutations /s": births add the parent's inherited
  `accumulatedMutations` to the sum, deaths remove the creature's total. In equilibrium these
  partially cancel but add noise. Cleaner alternative (recommendation, decide during
  implementation): an additional monotonic counter in `AccumulatedStatistics`
  (`totalMutations`), incremented in `updateAccumulatedMutationsAndLineageId` by
  `accumulatedMutations / denominator` — this measures genuine mutation events only.
  Requires threading `statistics` into the mutation kernel.
- Persistence: extend `ColumnDescriptions` + `getDataRef` in `SerializerService.cpp`
  additively. The CSV loader matches columns by name, so old `.statistics.csv` files load
  fine (new columns default to 0).
- `StatisticsWindow` remains unchanged.
- Dashboard header cards: "Sum energy" / "Creatures" / "Created" come from the data above.
  The "Entities" card (solids / fluid particles / cells / energy particles) needs trivial
  additional per-type counters in `TimestepStatistics` (switch on `object->type` in
  substep 2). "External energy" value availability via the facade to be checked (open
  question / scope).
- Dashboard wiring: `EvolutionDashboardWindow` gets an update path like
  `StatisticsWindow::processBackground` (or reuses a shared live-statistics helper):
  live mode from a `TimelineLiveStatistics`-style buffer, "Entire history" / "Last X steps"
  from `StatisticsHistory`.

## Phase 3 — per-lineage data (lineage table + per-lineage timelines)

Largest chunk, cleanly separable:

- GPU: hash map `lineageId → LineageStatistics` (pattern: existing
  `_mutantToMutantStatisticsMap` in `SimulationStatistics.cuh`). Per slot: `lineageId`,
  `colorBitset` (atomicOr of cell colors), `numCreatures`, `sumCells`, `sumNodes`,
  `sumEnergy`, `sumMutationRates`, `sumGenerations`, `sumAccumulatedMutations`.
  Reset per update.
- Per-lineage "Created /s": a second, *persistent* map (`lineageId → createdCount`, never
  reset), incremented at the `cloneCreature` site. Per-lineage "Mutations /s": GUI delta of
  `sumAccumulatedMutations` (same artifact note as above).
- Compaction kernel: write non-empty slots (count ≥ threshold) into a fixed-capacity output
  array (e.g. 1024 entries) with an atomic counter → small host copy per update, instead of
  copying the whole map.
- Host: new `LineageHistory` container analogous to `StatisticsHistory` (mutex, downsampling
  like `StatisticsService::addDataPoint`). Lineages appear and disappear → series keyed by
  `lineageId`, gaps allowed.
- GUI: replace `generateDummyData`; `DummyLineage` becomes a real structure fed from
  `LineageHistory` + live buffer. Filter/selection/plot code stays almost unchanged (same
  data shape).
- Persistence of lineage history: proposal — none initially ("Entire history" mode then only
  from session start); optionally later a separate file (e.g. `.dashboard.csv`).

## Suggested PR order

1. Phase 0 (refactoring, independently testable)
2. Phases 1 + 2 together — dashboard shows real global values; lineage table still shows
   only the "All lineages" row (or remaining dummy rows clearly marked)
3. Phase 3 (per-lineage)

## Open questions

1. "Avg mut. rate": plain mean over all 21 probability values, or weighted
   (e.g. node- vs. gene-level separately)?
2. Exclude creatures under construction (`numCells == 0`) from the counts?
3. Save-file migration as proposed (`accumulatedMutationsInLineage := old value`)?
4. "Mutations /s": sum-based (as proposed, with birth/death artifact) or additional
   monotonic `totalMutations` counter (cleaner, slightly more plumbing)?
5. Lineage cap / minimum-creature threshold for the table; persist lineage history?
6. "Entities" and "External energy" header cards in scope of the first iteration?
