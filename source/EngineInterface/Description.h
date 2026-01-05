#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <variant>

#include <Base/Macros.h>
#include <Base/MathTypes.h>

#include "Definitions.h"
#include "GenomeDescription.h"

struct ConnectionDescription
{
    auto operator<=>(ConnectionDescription const&) const = default;

    MEMBER(ConnectionDescription, uint64_t, cellId, 0);
    MEMBER(ConnectionDescription, float, distance, 0.0f);
    MEMBER(ConnectionDescription, float, angleFromPrevious, 0.0f);
};

struct StructureCellDescription
{
    auto operator<=>(StructureCellDescription const&) const = default;
};

struct FreeCellDescription
{
    auto operator<=>(FreeCellDescription const&) const = default;
};

struct NeuralNetworkDescription
{
    NeuralNetworkDescription();
    auto operator<=>(NeuralNetworkDescription const&) const = default;

    MEMBER(NeuralNetworkDescription, std::vector<float>, weights, {});
    MEMBER(NeuralNetworkDescription, std::vector<float>, biases, {});
    MEMBER(NeuralNetworkDescription, std::vector<ActivationFunction>, activationFunctions, {});

    NeuralNetworkDescription& weight(int row, int col, float value);
};

struct BaseDescription
{
    auto operator<=>(BaseDescription const&) const = default;
};

struct DepotDescription
{
    auto operator<=>(DepotDescription const&) const = default;

    MEMBER(DepotDescription, float, storageLimit, 200.0f);
    MEMBER(DepotDescription, float, storedUsableEnergy, 0.0f);
};

struct ConstructorDescription
{
    auto operator<=>(ConstructorDescription const&) const = default;

    // Properties
    MEMBER(ConstructorDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(ConstructorDescription, int, constructionActivationTime, 100);
    MEMBER(ConstructorDescription, float, constructionAngle, 0.0f);
    MEMBER(ConstructorDescription, ProvideEnergy, provideEnergy, ProvideEnergy_CellOnly);

    // Genome data
    MEMBER(ConstructorDescription, int, geneIndex, 0);

    // Process data
    MEMBER(ConstructorDescription, std::optional<uint64_t>, lastConstructedCellId, std::nullopt);
    MEMBER(ConstructorDescription, int, currentNodeIndex, 0);
    MEMBER(ConstructorDescription, int, currentConcatenation, 0);
    MEMBER(ConstructorDescription, int, currentBranch, 0);
};

struct TelemetryDescription
{
    auto operator<=>(TelemetryDescription const&) const = default;
};

struct DetectEnergyDescription
{
    auto operator<=>(DetectEnergyDescription const&) const = default;

    MEMBER(DetectEnergyDescription, float, minDensity, 0.05f);
};

struct DetectStructureDescription
{
    auto operator<=>(DetectStructureDescription const&) const = default;
};

struct DetectFreeCellDescription
{
    auto operator<=>(DetectFreeCellDescription const&) const = default;

    MEMBER(DetectFreeCellDescription, float, minDensity, 0.05f);
    MEMBER(DetectFreeCellDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct DetectCreatureDescription
{
    auto operator<=>(DetectCreatureDescription const&) const = default;

    MEMBER(DetectCreatureDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(DetectCreatureDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(DetectCreatureDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(DetectCreatureDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using SensorModeDescription = std::variant<TelemetryDescription, DetectEnergyDescription, DetectStructureDescription, DetectFreeCellDescription, DetectCreatureDescription>;

struct SensorLastMatchDescription
{
    auto operator<=>(SensorLastMatchDescription const&) const = default;

    MEMBER(SensorLastMatchDescription, uint64_t, creatureId, 0);
    MEMBER(SensorLastMatchDescription, RealVector2D, pos, RealVector2D());
};

struct SensorDescription
{
    auto operator<=>(SensorDescription const&) const = default;

    MEMBER(SensorDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(SensorDescription, SensorModeDescription, mode, DetectCreatureDescription());
    MEMBER(SensorDescription, int, minRange, 0);
    MEMBER(SensorDescription, int, maxRange, 255);

    // Process data
    MEMBER(SensorDescription, std::optional<SensorLastMatchDescription>, lastMatch, std::nullopt);

    SensorMode getMode() const;
};

struct GeneratorDescription
{
    auto operator<=>(GeneratorDescription const&) const = default;

    // Fixed data
    MEMBER(GeneratorDescription, int, autoTriggerInterval, 100);  // Must be >= 3
    MEMBER(GeneratorDescription, GeneratorPulseType, pulseType, GeneratorPulseType_Positive);
    MEMBER(
        GeneratorDescription,
        int,
        alternationInterval,
        20);  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    MEMBER(GeneratorDescription, int, numPulses, 0);
};

struct AttackFreeCellDescription
{
    auto operator<=>(AttackFreeCellDescription const&) const = default;

    MEMBER(AttackFreeCellDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct AttackCreatureDescription
{
    auto operator<=>(AttackCreatureDescription const&) const = default;

    MEMBER(AttackCreatureDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(AttackCreatureDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(AttackCreatureDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(AttackCreatureDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using AttackerModeDescription = std::variant<AttackFreeCellDescription, AttackCreatureDescription>;

struct AttackerDescription
{
    auto operator<=>(AttackerDescription const&) const = default;

    MEMBER(AttackerDescription, AttackerModeDescription, mode, AttackCreatureDescription());

    AttackerMode getMode() const;
};

struct InjectorDescription
{
    InjectorDescription();
    auto operator<=>(InjectorDescription const&) const = default;

    MEMBER(InjectorDescription, int, geneIndex, 0);
};

struct AutoBendingDescription
{
    auto operator<=>(AutoBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoBendingDescription, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(AutoBendingDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AutoBendingDescription, std::optional<float>, initialAngle, std::nullopt);
    MEMBER(AutoBendingDescription, bool, forward, true);  // Current direction
    MEMBER(AutoBendingDescription, float, activation, 0);
    MEMBER(AutoBendingDescription, int, activationCountdown, 0);
    MEMBER(AutoBendingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualBendingDescription
{
    auto operator<=>(ManualBendingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualBendingDescription, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(ManualBendingDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(ManualBendingDescription, std::optional<float>, initialAngle, std::nullopt);
    MEMBER(ManualBendingDescription, float, lastAngleDelta, 0.0f);
    MEMBER(ManualBendingDescription, bool, impulseAlreadyApplied, false);
};

struct AngleBendingDescription
{
    auto operator<=>(AngleBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AngleBendingDescription, float, maxAngleDeviation, 0.2f);         // Between 0 and 1
    MEMBER(AngleBendingDescription, float, attractionRepulsionRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AngleBendingDescription, std::optional<float>, initialAngle, std::nullopt);
};

struct AutoCrawlingDescription
{
    auto operator<=>(AutoCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoCrawlingDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(AutoCrawlingDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AutoCrawlingDescription, std::optional<float>, initialDistance, std::nullopt);
    MEMBER(AutoCrawlingDescription, float, lastActualDistance, 0.0f);
    MEMBER(AutoCrawlingDescription, bool, forward, true);  // Current direction
    MEMBER(AutoCrawlingDescription, float, activation, 0.0f);
    MEMBER(AutoCrawlingDescription, int, activationCountdown, 0);
    MEMBER(AutoCrawlingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualCrawlingDescription
{
    auto operator<=>(ManualCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualCrawlingDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(ManualCrawlingDescription, std::optional<float>, initialDistance, std::nullopt);
    MEMBER(ManualCrawlingDescription, float, lastActualDistance, 0.0f);
    MEMBER(ManualCrawlingDescription, float, lastDistanceDelta, 0.0f);
    MEMBER(ManualCrawlingDescription, bool, impulseAlreadyApplied, false);
};

struct DirectMovementDescription
{
    auto operator<=>(DirectMovementDescription const&) const = default;
};

using MuscleModeDescription = std::variant<
    AutoBendingDescription,
    ManualBendingDescription,
    AngleBendingDescription,
    AutoCrawlingDescription,
    ManualCrawlingDescription,
    DirectMovementDescription>;

struct MuscleDescription
{
    auto operator<=>(MuscleDescription const&) const = default;

    MEMBER(MuscleDescription, MuscleModeDescription, mode, AutoBendingDescription());

    // Additional rendering data
    MEMBER(MuscleDescription, float, lastMovementX, 0.0f);
    MEMBER(MuscleDescription, float, lastMovementY, 0.0f);

    MuscleMode getMode() const;
};

struct DefenderDescription
{
    auto operator<=>(DefenderDescription const&) const = default;

    MEMBER(DefenderDescription, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectStructureDescription
{
    auto operator<=>(ReconnectStructureDescription const&) const = default;
};

struct ReconnectFreeCellDescription
{
    auto operator<=>(ReconnectFreeCellDescription const&) const = default;

    MEMBER(ReconnectFreeCellDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct ReconnectCreatureDescription
{
    auto operator<=>(ReconnectCreatureDescription const&) const = default;

    MEMBER(ReconnectCreatureDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(ReconnectCreatureDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(ReconnectCreatureDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectCreatureDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using ReconnectorModeDescription = std::variant<ReconnectStructureDescription, ReconnectFreeCellDescription, ReconnectCreatureDescription>;

struct ReconnectorDescription
{
    auto operator<=>(ReconnectorDescription const&) const = default;

    MEMBER(ReconnectorDescription, ReconnectorModeDescription, mode, ReconnectCreatureDescription());

    ReconnectorMode getMode() const;
};

struct DetonatorDescription
{
    auto operator<=>(DetonatorDescription const&) const = default;

    MEMBER(DetonatorDescription, DetonatorState, state, DetonatorState_Ready);
    MEMBER(DetonatorDescription, int, countdown, 60);
};

struct DigestorDescription
{
    auto operator<=>(DigestorDescription const&) const = default;

    MEMBER(DigestorDescription, float, rawEnergyConductivity, 0.5f);    // Between 0 and 1

    float getRawEnergyConversionRate() const { return 1 - _rawEnergyConductivity; }
    DigestorDescription& setRawEnergyConversionRate(float value)
    {
        _rawEnergyConductivity = 1 - value;
        return *this;
    }
};

struct SignalEntryDescription
{
    SignalEntryDescription();
    auto operator<=>(SignalEntryDescription const&) const = default;

    MEMBER(SignalEntryDescription, std::vector<float>, channels, {});
};

struct SignalDelayDescription
{
    auto operator<=>(SignalDelayDescription const&) const = default;

    MEMBER(SignalDelayDescription, int, delay, 10);
    MEMBER(SignalDelayDescription, int, numSignalEntriesInitialized, 0);
    MEMBER(SignalDelayDescription, int, ringBufferIndex, 0);
};

struct SignalRecorderDescription
{
    auto operator<=>(SignalRecorderDescription const&) const = default;

    MEMBER(SignalRecorderDescription, bool, readOnly, true);
    MEMBER(SignalRecorderDescription, SignalRecorderState, state, SignalRecorderState_Idle);
    MEMBER(SignalRecorderDescription, int, numWrittenSignalEntries, 0);
    MEMBER(SignalRecorderDescription, int, numReadSignalEntries, 0);
};

struct SignalStorageDescription
{
    auto operator<=>(SignalStorageDescription const&) const = default;

    MEMBER(SignalStorageDescription, bool, readOnly, true);
};

struct SignalIntegratorDescription
{
    auto operator<=>(SignalIntegratorDescription const&) const = default;

    MEMBER(SignalIntegratorDescription, float, newSignalWeight, 0.5f);  // Between 0 and 1
};

using MemoryModeDescription = std::variant<SignalDelayDescription, SignalRecorderDescription, SignalStorageDescription, SignalIntegratorDescription>;

struct MemoryDescription
{
    auto operator<=>(MemoryDescription const&) const = default;

    MEMBER(MemoryDescription, MemoryModeDescription, mode, SignalDelayDescription());
    MEMBER(MemoryDescription, std::vector<SignalEntryDescription>, signalEntries, {});
    MEMBER(MemoryDescription, uint8_t, channelBitMask, 0b11111111);

    MemoryMode getMode() const;
};

using CellTypeDescription = std::variant<
    StructureCellDescription,
    FreeCellDescription,
    BaseDescription,
    DepotDescription,
    ConstructorDescription,
    SensorDescription,
    GeneratorDescription,
    AttackerDescription,
    InjectorDescription,
    MuscleDescription,
    DefenderDescription,
    ReconnectorDescription,
    DetonatorDescription,
    DigestorDescription,
    MemoryDescription>;

struct SignalRestrictionDescription
{
    auto operator<=>(SignalRestrictionDescription const&) const = default;

    MEMBER(SignalRestrictionDescription, SignalRestrictionMode, mode, SignalRestrictionMode_Inactive);
    MEMBER(SignalRestrictionDescription, float, baseAngle, 0);
    MEMBER(SignalRestrictionDescription, float, openingAngle, 90.0f);
};

struct SignalDescription
{
    SignalDescription();
    auto operator<=>(SignalDescription const&) const = default;

    MEMBER(SignalDescription, std::vector<float>, channels, {});
};

struct CellDescription
{
    CellDescription(bool createIds = true);
    auto operator<=>(CellDescription const&) const = default;

    // General
    uint64_t _id = 0;
    CellDescription id(uint64_t id);
    MEMBER(CellDescription, std::vector<ConnectionDescription>, connections, {});
    MEMBER(CellDescription, RealVector2D, pos, RealVector2D());
    MEMBER(CellDescription, RealVector2D, vel, RealVector2D());
    MEMBER(CellDescription, float, usableEnergy, 100.0f);
    MEMBER(CellDescription, float, rawEnergy, 0.0f);
    MEMBER(CellDescription, float, stiffness, 1.0f);
    MEMBER(CellDescription, int, color, 0);
    MEMBER(
        CellDescription,
        std::optional<float>,
        frontAngle,
        std::nullopt);  // Angle between [cell, cell->connection[0]] and front direction in reference configuration
    MEMBER(CellDescription, bool, fixed, false);
    MEMBER(CellDescription, bool, sticky, false);
    MEMBER(CellDescription, int, age, 0);
    MEMBER(CellDescription, CellState, cellState, CellState_Ready);

    // Creature/genome data
    MEMBER(CellDescription, int, nodeIndex, 0);
    MEMBER(CellDescription, int, parentNodeIndex, 0);
    MEMBER(CellDescription, int, geneIndex, 0);

    // Cell type-specific data
    MEMBER(CellDescription, std::optional<NeuralNetworkDescription>, neuralNetwork, std::nullopt);
    MEMBER(CellDescription, CellTypeDescription, cellType, BaseDescription());
    MEMBER(CellDescription, SignalState, signalState, SignalState_Inactive);
    MEMBER(CellDescription, SignalDescription, signal, SignalDescription());    // For signalState == SignalState_Active
    MEMBER(CellDescription, SignalRestrictionDescription, signalRestriction, SignalRestrictionDescription());
    MEMBER(CellDescription, int, activationTime, 0);
    MEMBER(CellDescription, CellTriggered, cellTriggered, CellTriggered_No);

    // Process data
    MEMBER(CellDescription, int, frontAngleId, 0);
    MEMBER(CellDescription, bool, headCell, false);

    CellType getCellType() const;
    CellDescription& signalAndState(std::vector<float> const& value);
    CellDescription& signalRestriction(float baseAngle, float openingAngle);

    bool isConnectedTo(uint64_t id) const;
    float getAngleSpan(uint64_t connectedCellId1, uint64_t connectedCellId2) const;
};

struct ParticleDescription
{
    ParticleDescription();
    auto operator<=>(ParticleDescription const&) const = default;

    uint64_t _id = 0;
    ParticleDescription id(uint64_t id);
    MEMBER(ParticleDescription, RealVector2D, pos, RealVector2D());
    MEMBER(ParticleDescription, RealVector2D, vel, RealVector2D());
    MEMBER(ParticleDescription, float, energy, 0.0f);
    MEMBER(ParticleDescription, int, color, 0);
};

struct CreatureDescription
{
    CreatureDescription();
    auto operator<=>(CreatureDescription const&) const = default;

    uint64_t _id = 0;
    CreatureDescription id(uint64_t id);
    MEMBER(CreatureDescription, std::optional<uint64_t>, ancestorId, std::nullopt);
    MEMBER(CreatureDescription, int, generation, 0);
    MEMBER(CreatureDescription, int, lineageId, 0);
    MEMBER(CreatureDescription, int, numCells, 0);
    MEMBER(CreatureDescription, uint64_t, genomeId, 0);
    MEMBER(CreatureDescription, std::vector<CellDescription>, cells, {});

    // Process data
    MEMBER(CreatureDescription, int, frontAngleId, 0);
};

struct _DescriptionCache
{
    struct Index
    {
        std::optional<int> creatureIndex;
        int cellIndex;
    };
    std::unordered_map<uint64_t, Index> cellIdToIndex;
    std::unordered_map<uint64_t, uint64_t> genomeIdToIndex;
};
using DescriptionCache = std::shared_ptr<_DescriptionCache>;

struct Description
{
    auto operator<=>(Description const&) const = default;

    MEMBER(Description, std::vector<CellDescription>, cells, {});
    MEMBER(Description, std::vector<ParticleDescription>, particles, {});
    MEMBER(Description, std::vector<CreatureDescription>, creatures, {});
    MEMBER(Description, std::vector<GenomeDescription>, genomes, {});

    void forEachCell(std::function<void(CellDescription const&)> const& applyFunc) const;
    void forEachCell(std::function<void(CellDescription&)> const& applyFunc);
    // First parameter of lambda is creature index, second parameter is cell index
    void forEachCell(std::function<void(std::optional<uint64_t> const&, uint64_t, CellDescription const&)> const& applyFunc) const;
    void forEachCell(std::function<void(std::optional<uint64_t> const&, uint64_t, CellDescription&)> const& applyFunc);

    size_t getNumCells() const;

    void clear();
    bool isEmpty() const;

    Description& add(Description&& other, bool assignNewIds = true);

    bool hasUniqueIds() const;
    void assignNewIds();  // Preserves order of cell ids

    Description& addCreature(CreatureDescription const& creature, GenomeDescription const& genome = GenomeDescription());

    DescriptionCache createCache() const;
    Description& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, DescriptionCache const& cache = nullptr);
    Description& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, RealVector2D const& refPosCell2, DescriptionCache const& cache = nullptr);

    CellDescription const& getCellRef(uint64_t const& cellId, DescriptionCache const& cache = nullptr) const;
    CellDescription& getCellRef(uint64_t const& cellId, DescriptionCache const& cache = nullptr);
    CellDescription& getCellRef(std::optional<uint64_t> const& creatureIndex, uint64_t const& cellIndex);

    CellDescription& getOtherCellRef(uint64_t id);
    CellDescription& getOtherCellRef(std::set<uint64_t> const& ids);
    std::vector<CellDescription> getOtherCells(std::set<uint64_t> const& ids) const;

    GenomeDescription const& getGenomeRef(uint64_t const& genomeId, DescriptionCache const& cache = nullptr) const;

    bool hasConnection(uint64_t id, uint64_t otherId) const;
    bool hasConnection(CellDescription const& cell1, CellDescription const& cell2) const;
    ConnectionDescription& getConnectionRef(uint64_t id, uint64_t otherId);
    ConnectionDescription const& getConnection(CellDescription const& cell1, CellDescription const& cell2) const;
    CreatureDescription& getCreatureRef(uint64_t id);
    CreatureDescription& getOtherCreatureRef(uint64_t id);

private:
    _DescriptionCache::Index getCellIndex(uint64_t const& cellId, DescriptionCache const& cache) const;
};

struct ExtendedCellDescription
{
    CellDescription cell;
    std::optional<uint64_t> creatureId;
    std::optional<GenomeDescription> genome;
};
using ExtendedCellOrParticleDescription = std::variant<ExtendedCellDescription, ParticleDescription>;
