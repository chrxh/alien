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

### The core problem: dimension change from color to active lineages

This is the biggest structural change and drives the whole design. The existing statistics
are keyed by **color** — a fixed dimension of `MAX_COLORS = 10`. Everything downstream
relies on that: `ColorVector<T>` in `StatisticsRawData`, `DataPoint::values[MAX_COLORS]`,
the fixed CSV column set, the plot code. The dashboard instead keys by **active lineage**,
where *active* means: at least one living creature carries this `lineageId`. That dimension
is fundamentally different:

- **Dynamic and unbounded**: lineage ids come from an atomic counter
  (`CudaNumberGenerator::createLineageId`); new lineages appear (threshold splits, seeds)
  and disappear (last creature dies). Directly after seeding, the number of active lineages
  can be on the order of the creature count.
- **Not enumerable directly**: there is no global creature or lineage array on the GPU;
  creatures are only reachable via `Cell::creature`. The set of active lineages must first
  be *discovered* each collection cycle before anything can be aggregated per lineage.
- **No fixed-size container fits**: none of the existing structures
  (`ColorVector`, `DataPoint`, CSV columns) can represent a per-lineage breakdown. Both the
  GPU side and the host/history side need new, dynamically sized data paths.

Consequently the collection is a two-stage problem, exactly as it has to be solved on the
GPU: (1) determine all active lineages, (2) accumulate the statistics for these lineages.

### GPU design: discovery + aggregation via exact hash map

New device structure: open-addressing hash map keyed by `lineageId` (key stored in the
slot, insert via `atomicCAS`, linear probing). Note: the existing
`_mutantToMutantStatisticsMap` is *not* a suitable pattern — it merges colliding ids into
one bucket (`lineageId % size`), which is fine for a diversity estimate but wrong for a
table with exact per-lineage rows.

Per slot: `lineageId` (key, 0 = empty), `colorBitset`, `numCreatures`, `sumCells`,
`sumNodes`, `sumEnergy`, `sumMutationRates`, `sumGenerations`, `sumAccumulatedMutations`.

Collection cycle (new substeps in `StatisticsKernels.cu`; they run inside
`updateStatistics`, which is throttled to one execution per 30 ms wall clock
(`StatisticsUpdate` in `SimulationCudaFacade.cu`), *not* per timestep — so the extra cost
is bounded):

1. **Clear**: reset all slot keys/values (plain memset-style kernel over the map).
2. **Discover lineages + creature-level aggregates**: iterate objects; deduplicate per
   creature via `atomicExch` sentinel on `creature->creatureIndex`
   (`DataAccessKernels.cu` pattern). The first thread per creature does
   `slot = insertOrFind(creature->lineageId)` and atomically adds `numCreatures`,
   `sumCells`, `sumGenerations`, `sumAccumulatedMutations`. It then **stores the slot index
   in `creature->creatureIndex`** — all later passes reach the slot in O(1) without
   re-probing the hash map.
3. **Genome-level aggregates**: deduplicate per genome via `genome->genomeIndex`; add
   `sumNodes` and `sumMutationRates` to the owning creature's cached slot. (Genome sharing
   only occurs between a parent and its not-yet-mutated offspring — see
   `ConstructorProcessor::mutateGenome` — which by construction have the same `lineageId`,
   so the attribution is unambiguous.)
4. **Cell-level aggregates**: every cell adds its energy and `atomicOr`s its color into
   `colorBitset`, using the O(1) cached slot from `creature->creatureIndex`.
5. **Compaction**: scan the map; every slot with `numCreatures ≥ 1` is written into a
   compact output array via an atomic counter. This also yields `numActiveLineages`
   (header KPI "LINEAGES") for free. Only the compact array is copied to the host —
   not the map.

Sizing and overflow: map capacity is a fixed power of two (e.g. 2^18 slots ≈ 262k
lineages at ~40–64 B/slot ≈ 10–16 MB device memory; target load factor ≤ 0.5). If more
lineages are active than the map can hold, further inserts are dropped and an overflow flag
is set (GUI shows a warning). The capacity must be chosen relative to realistic creature
counts — open question.

Per-lineage "Created /s" needs a *persistent* accumulated counter per lineage
(`lineageId → createdCount`, incremented at the `cloneCreature` site, never cleared).
A persistent map fills up with dead lineages over time, so it needs occasional garbage
collection (evict ids that have been inactive for N cycles, using the activity information
from step 5). Alternative for the first iteration: only the global "Created /s" (Phase 2)
and no per-lineage value in the table — open question.

### Host-side data model and history

`DataPointCollection` cannot be reused for per-lineage data (fixed struct, arithmetic
operators, CSV coupling). New parallel path:

- `LineageStatisticsEntry { lineageId, colorBitset, numCreatures, sums... }`;
  one sample = `{ timestep, systemClock, std::vector<LineageStatisticsEntry> }`.
- Transport: separate device-to-host copy of the compact array, exposed next to
  `getStatisticsRawData()` (keeping `StatisticsRawData` itself a fixed-size memcpy).
- `LineageHistory` analogous to `StatisticsHistory` (mutex, downsampling like
  `StatisticsService::addDataPoint`). Lineages come and go, so series are keyed by
  `lineageId` **with gaps**. Downsampling/merging two samples = union of the id sets,
  merging values where an id is present in both. This is the main added complexity vs.
  the color-keyed history and should get its own unit tests.
- Rates per lineage ("Created /s", "Mutations /s") are computed GUI-side from deltas of the
  accumulated sums between two samples in which the lineage is present; no rate point when
  it is absent from either sample.
- Table: latest sample, sorted by `numCreatures` descending, displayed rows capped
  (threshold or top-K; `numActiveLineages` still shows the true total). "All lineages" row
  comes from the global Phase 1/2 data.
- Selection and color filter: keyed by `lineageId` / `colorBitset` — extinct lineages
  disappear from the table but may remain visible in historical timelines.
- GUI: replace `generateDummyData`; `DummyLineage` becomes the real structure fed from
  `LineageHistory` + live buffer. Filter/selection/plot code stays almost unchanged (same
  data shape).

### Persistence

The `.statistics.csv` format (fixed, name-matched columns) cannot represent a dynamic
lineage dimension. Proposal: no persistence of the lineage history in the first iteration
("Entire history" mode then covers the time since session start / file load); later
optionally a separate long-format file (one row per `(time, lineageId)`, e.g.
`.lineages.csv`).

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
7. Hash-map capacity for active lineages (memory vs. worst case right after seeding, where
   #lineages ≈ #creatures)?
8. Per-lineage "Created /s" in the first iteration (needs persistent per-lineage counter
   map + GC) or global only?
