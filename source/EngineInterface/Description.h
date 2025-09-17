#pragma once

#include <variant>
#include <optional>
#include <string>
#include <unordered_set>

#include "Base/Macros.h"
#include "Base/Vector2D.h"

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

    MEMBER(DepotDescription, EnergyDistributionMode, mode, EnergyDistributionMode_TransmittersAndConstructors);
};

struct ConstructorDescription
{
    auto operator<=>(ConstructorDescription const&) const = default;

    // Properties
    MEMBER(ConstructorDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(ConstructorDescription, int, constructionActivationTime, 100);
    MEMBER(ConstructorDescription, float, constructionAngle, 0.0f);

    // Genome data
    MEMBER(ConstructorDescription, int, geneIndex, 0);

    // Process data
    MEMBER(ConstructorDescription, std::optional<uint64_t>, lastConstructedCellId, std::nullopt);
    MEMBER(ConstructorDescription, int, currentNodeIndex, 0);
    MEMBER(ConstructorDescription, int, currentConcatenation, 0);
    MEMBER(ConstructorDescription, int, currentBranch, 0);
};

struct SensorDescription
{
    auto operator<=>(SensorDescription const&) const = default;

    MEMBER(SensorDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(SensorDescription, float, minDensity, 0.05f);
    MEMBER(SensorDescription, std::optional<int>, minRange, std::nullopt);
    MEMBER(SensorDescription, std::optional<int>, maxRange, std::nullopt);
    MEMBER(SensorDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(SensorDescription, SensorRestrictToCreatures, restrictToCreatures, SensorRestrictToCreatures_NoRestriction);
};

struct GeneratorDescription
{
    auto operator<=>(GeneratorDescription const&) const = default;

    // Fixed data
    MEMBER(GeneratorDescription, int, autoTriggerInterval, 100);   // Must be >= 3
    MEMBER(GeneratorDescription, GeneratorPulseType, pulseType, GeneratorPulseType_Positive);
    MEMBER(GeneratorDescription, int, alternationInterval, 1);  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    MEMBER(GeneratorDescription, int, numPulses, 0);
};

struct AttackerDescription
{
    auto operator<=>(AttackerDescription const&) const = default;
};

struct InjectorDescription
{
    InjectorDescription();
    auto operator<=>(InjectorDescription const&) const = default;

    MEMBER(InjectorDescription, InjectorMode, mode, InjectorMode_InjectAll);
    MEMBER(InjectorDescription, int, counter, 0);
};

struct AutoBendingDescription
{
    auto operator<=>(AutoBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoBendingDescription, float, maxAngleDeviation, 0.2f);    // Between 0 and 1
    MEMBER(AutoBendingDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1

    // Process data
    MEMBER(AutoBendingDescription, float, initialAngle, 0.0f);
    MEMBER(AutoBendingDescription, float, lastActualAngle, 0.0f);
    MEMBER(AutoBendingDescription, bool, forward, true);    // Current direction
    MEMBER(AutoBendingDescription, float, activation, 0);
    MEMBER(AutoBendingDescription, int, activationCountdown, 0);
    MEMBER(AutoBendingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualBendingDescription
{
    auto operator<=>(ManualBendingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualBendingDescription, float, maxAngleDeviation, 0.2f);   // Between 0 and 1
    MEMBER(ManualBendingDescription, float, frontBackVelRatio, 0.2f);   // Between 0 and 1

    // Process data
    MEMBER(ManualBendingDescription, float, initialAngle, 0.0f);
    MEMBER(ManualBendingDescription, float, lastActualAngle, 0.0f);
    MEMBER(ManualBendingDescription, float, lastAngleDelta, 0.0f);
    MEMBER(ManualBendingDescription, bool, impulseAlreadyApplied, false);
};

struct AngleBendingDescription
{
    auto operator<=>(AngleBendingDescription const&) const = default;

    // Fixed data
    MEMBER(AngleBendingDescription, float, maxAngleDeviation, 0.2f);   // Between 0 and 1
    MEMBER(AngleBendingDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1

    // Process data
    MEMBER(AngleBendingDescription, float, initialAngle, 0.0f);
};

struct AutoCrawlingDescription
{
    auto operator<=>(AutoCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(AutoCrawlingDescription, float, maxDistanceDeviation, 0.8f); // Between 0 and 1
    MEMBER(AutoCrawlingDescription, float, frontBackVelRatio, 0.2f);    // Between 0 and 1

    // Process data
    MEMBER(AutoCrawlingDescription, float, initialDistance, 0.0f);
    MEMBER(AutoCrawlingDescription, float, lastActualDistance, 0.0f);
    MEMBER(AutoCrawlingDescription, bool, forward, true);               // Current direction
    MEMBER(AutoCrawlingDescription, float, activation, 0.0f);
    MEMBER(AutoCrawlingDescription, int, activationCountdown, 0);
    MEMBER(AutoCrawlingDescription, bool, impulseAlreadyApplied, false);
};

struct ManualCrawlingDescription
{
    auto operator<=>(ManualCrawlingDescription const&) const = default;

    // Fixed data
    MEMBER(ManualCrawlingDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingDescription, float, frontBackVelRatio, 0.2f);   // Between 0 and 1

    // Process data
    MEMBER(ManualCrawlingDescription, float, initialDistance, 0.0f);
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

    MEMBER(MuscleDescription, MuscleModeDescription, mode, MuscleModeDescription());

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

struct ReconnectorDescription
{
    auto operator<=>(ReconnectorDescription const&) const = default;

    MEMBER(ReconnectorDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectorDescription, ReconnectorRestrictToCreatures, restrictToCreatures, ReconnectorRestrictToCreatures_NoRestriction);
};

struct DetonatorDescription
{
    auto operator<=>(DetonatorDescription const&) const = default;

    MEMBER(DetonatorDescription, DetonatorState, state, DetonatorState_Ready);
    MEMBER(DetonatorDescription, int, countdown, 60);
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
    DetonatorDescription>;

struct SignalRestrictionDescription
{
    auto operator<=>(SignalRestrictionDescription const&) const = default;

    MEMBER(SignalRestrictionDescription, bool, active, false);
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
    MEMBER(CellDescription, float, energy, 100.0f);
    MEMBER(CellDescription, float, stiffness, 1.0f);
    MEMBER(CellDescription, int, color, 0);
    MEMBER(CellDescription, std::optional<float>, frontAngle, std::nullopt); // Angle between [cell, cell->connection[0]] and front direction in reference configuration
    MEMBER(CellDescription, bool, barrier, false);
    MEMBER(CellDescription, bool, sticky, false);
    MEMBER(CellDescription, int, age, 0);
    MEMBER(CellDescription, CellState, cellState, CellState_Ready);

    // Creature/genome data
    MEMBER(CellDescription, int, nodeIndex, 0);
    MEMBER(CellDescription, int, parentNodeIndex, 0);
    MEMBER(CellDescription, int, geneIndex, 0);

    // Cell type-specific data
    MEMBER(CellDescription, std::optional<NeuralNetworkDescription>, neuralNetwork, std::nullopt);
    MEMBER(CellDescription, CellTypeDescription, cellTypeData, BaseDescription());
    MEMBER(CellDescription, SignalRestrictionDescription, signalRestriction, SignalRestrictionDescription());
    MEMBER(CellDescription, uint8_t, signalRelaxationTime, 0);
    MEMBER(CellDescription, std::optional<SignalDescription>, signal, std::nullopt);
    MEMBER(CellDescription, int, activationTime, 0);
    MEMBER(CellDescription, int, detectedByCreatureId, 0);  // Only the first 16 bits from the creature id
    MEMBER(CellDescription, CellTriggered, cellTriggered, CellTriggered_No);

    // Process data
    MEMBER(CellDescription, int, frontAngleId, 0);
    MEMBER(CellDescription, bool, isFrontAngleRefCell, false);

    CellType getCellType() const;
    CellDescription& signalAndRelaxTime(std::vector<float> const& value);
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
    MEMBER(CreatureDescription, GenomeDescription, genome, {});
    MEMBER(CreatureDescription, std::vector<CellDescription>, cells, {});

    // Process data
    MEMBER(CreatureDescription, int, frontAngleId, 0);
};

struct _CollectionCache
{
    struct Index
    {
        std::optional<int> creatureIndex;
        int cellIndex;
    };
    std::unordered_map<uint64_t, Index> cellIdToIndex;
};
using CollectionCache = std::shared_ptr<_CollectionCache>;

struct Description
{
    auto operator<=>(Description const&) const = default;

    MEMBER(Description, std::vector<CellDescription>, cells, {});
    MEMBER(Description, std::vector<ParticleDescription>, particles, {});
    MEMBER(Description, std::vector<CreatureDescription>, creatures, {});

    void forEachCell(std::function<void(CellDescription const&)> const& applyFunc) const;
    void forEachCell(std::function<void(CellDescription&)> const& applyFunc);
    // First parameter of lambda is creature index, second parameter is cell index
    void forEachCell(std::function<void(std::optional<uint64_t> const&, uint64_t, CellDescription const&)> const& applyFunc) const;
    void forEachCell(std::function<void(std::optional<uint64_t> const&, uint64_t, CellDescription&)> const& applyFunc);  

    void clear();
    bool isEmpty() const;

    void add(Description&& other);

    bool hasUniqueIds() const;
    void assignNewIds();  // Preserves order of cell ids

    CollectionCache createCache() const;
    Description& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, CollectionCache const& cache = nullptr);
    Description& addConnection(uint64_t const& cellId1, uint64_t const& cellId2, RealVector2D const& refPosCell2, CollectionCache const& cache = nullptr);
    CellDescription const& getCellRef(uint64_t const& cellId, CollectionCache const& cache = nullptr) const;
    CellDescription& getCellRef(uint64_t const& cellId, CollectionCache const& cache = nullptr);
    CellDescription& getOtherCellRef(uint64_t id);
    CellDescription& getOtherCellRef(std::set<uint64_t> const& ids);
    std::vector<CellDescription> getOtherCells(std::set<uint64_t> const& ids) const;
    CellDescription& getCellRef(std::optional<uint64_t> const& creatureIndex, uint64_t const& cellIndex);
    bool hasConnection(uint64_t id, uint64_t otherId) const;
    bool hasConnection(CellDescription const& cell1, CellDescription const& cell2) const;
    ConnectionDescription getConnection(uint64_t id, uint64_t otherId) const;
    ConnectionDescription getConnection(CellDescription const& cell1, CellDescription const& cell2) const;
    CreatureDescription& getCreatureRef(uint64_t id);
    CreatureDescription& getOtherCreatureRef(uint64_t id);

private:
    _CollectionCache::Index getCellIndex(uint64_t const& cellId, CollectionCache const& cache) const;
};

struct ExtendedCellDescription
{
    CellDescription cell;
    std::optional<uint64_t> creatureId;
    std::optional<GenomeDescription> genome;
};
using ExtendedCellOrParticleDescription = std::variant<ExtendedCellDescription, ParticleDescription>;
