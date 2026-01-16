#pragma once

#include <algorithm>
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

    MEMBER(ConnectionDescription, uint64_t, objectId, 0);
    MEMBER(ConnectionDescription, float, distance, 0.0f);
    MEMBER(ConnectionDescription, float, angleFromPrevious, 0.0f);
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

struct SenderDescription
{
    auto operator<=>(SenderDescription const&) const = default;

    MEMBER(SenderDescription, float, range, 15.0f);
    MEMBER(SenderDescription, int, maxTimesSent, 4);
};

struct ReceiverDescription
{
    auto operator<=>(ReceiverDescription const&) const = default;

    MEMBER(ReceiverDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReceiverDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using CommunicatorModeDescription = std::variant<SenderDescription, ReceiverDescription>;

struct CommunicatorDescription
{
    auto operator<=>(CommunicatorDescription const&) const = default;

    MEMBER(CommunicatorDescription, CommunicatorModeDescription, mode, SenderDescription());

    CommunicatorMode getMode() const;
};

using CellTypeDescription = std::variant<
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
    MemoryDescription,
    CommunicatorDescription>;

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
    MEMBER(SignalDescription, int, numTimesSent, 0);
};

struct StructureDescription
{
    auto operator<=>(StructureDescription const&) const = default;
};

struct FreeCellDescription
{
    auto operator<=>(FreeCellDescription const&) const = default;

    MEMBER(FreeCellDescription, float, rawEnergy, 100.0f);
};

struct CellDescription
{
    auto operator<=>(CellDescription const&) const = default;

    MEMBER(CellDescription, float, usableEnergy, 100.0f);
    MEMBER(CellDescription, float, rawEnergy, 0.0f);
    MEMBER(
        CellDescription,
        std::optional<float>,
        frontAngle,
        std::nullopt);  // Angle between [cell, cell->connection[0]] and front direction in reference configuration
    MEMBER(CellDescription, int, age, 0);
    MEMBER(CellDescription, CellState, cellState, CellState_Ready);

    // Creature/genome data
    MEMBER(CellDescription, std::optional<uint64_t>, creatureId, std::nullopt);
    MEMBER(CellDescription, int, nodeIndex, 0);
    MEMBER(CellDescription, int, parentNodeIndex, 0);
    MEMBER(CellDescription, int, geneIndex, 0);

    // Cell type-specific data
    MEMBER(CellDescription, std::optional<NeuralNetworkDescription>, neuralNetwork, std::nullopt);
    MEMBER(CellDescription, CellTypeDescription, cellType, BaseDescription());
    MEMBER(CellDescription, SignalState, signalState, SignalState_Inactive);
    MEMBER(CellDescription, SignalDescription, signal, SignalDescription());  // For signalState == SignalState_Active
    MEMBER(CellDescription, SignalRestrictionDescription, signalRestriction, SignalRestrictionDescription());
    MEMBER(CellDescription, int, activationTime, 0);
    MEMBER(CellDescription, CellTriggered, cellTriggered, CellTriggered_No);

    // Process data
    MEMBER(CellDescription, int, frontAngleId, 0);
    MEMBER(CellDescription, bool, headCell, false);

    CellType getCellType() const;
    CellDescription& signalAndState(std::vector<float> const& value);
    CellDescription& signalRestriction(float baseAngle, float openingAngle);
};

using ObjectTypeDescription = std::variant<StructureDescription, FreeCellDescription, CellDescription>;

struct ObjectDescription
{
    ObjectDescription(bool createIds = true);
    auto operator<=>(ObjectDescription const&) const = default;

    uint64_t _id = 0;
    ObjectDescription id(uint64_t id);
    MEMBER(ObjectDescription, std::vector<ConnectionDescription>, connections, {});
    MEMBER(ObjectDescription, RealVector2D, pos, RealVector2D());
    MEMBER(ObjectDescription, RealVector2D, vel, RealVector2D());
    MEMBER(ObjectDescription, float, stiffness, 1.0f);
    MEMBER(ObjectDescription, int, color, 0);
    MEMBER(ObjectDescription, bool, fixed, false);
    MEMBER(ObjectDescription, bool, sticky, false);
    MEMBER(ObjectDescription, ObjectTypeDescription, type, CellDescription());

    ObjectType getObjectType() const;

    FreeCellDescription& getFreeCellRef();
    FreeCellDescription const& getFreeCellRef() const;
    CellDescription& getCellRef();
    CellDescription const& getCellRef() const;

    bool isConnectedTo(uint64_t id) const;
    float getAngleSpan(uint64_t connectedObjectId1, uint64_t connectedObjectId2) const;
};

struct EnergyDescription
{
    EnergyDescription();
    auto operator<=>(EnergyDescription const&) const = default;

    uint64_t _id = 0;
    EnergyDescription id(uint64_t id);
    MEMBER(EnergyDescription, RealVector2D, pos, RealVector2D());
    MEMBER(EnergyDescription, RealVector2D, vel, RealVector2D());
    MEMBER(EnergyDescription, float, energy, 0.0f);
    MEMBER(EnergyDescription, int, color, 0);
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
    MEMBER(CreatureDescription, int, numObjects, 0);
    MEMBER(CreatureDescription, uint64_t, genomeId, 0);

    // Process data
    MEMBER(CreatureDescription, int, frontAngleId, 0);
};

struct _DescriptionCache
{
    std::unordered_map<uint64_t, int> objectIdToIndex;
    std::unordered_map<uint64_t, uint64_t> creatureIdToIndex;
    std::unordered_map<uint64_t, uint64_t> genomeIdToIndex;
};
using DescriptionCache = std::shared_ptr<_DescriptionCache>;

struct Description
{
    auto operator<=>(Description const&) const = default;

    MEMBER(Description, std::vector<ObjectDescription>, objects, {});
    MEMBER(Description, std::vector<EnergyDescription>, energies, {});
    MEMBER(Description, std::vector<CreatureDescription>, creatures, {});
    MEMBER(Description, std::vector<GenomeDescription>, genomes, {});

    void clear();
    bool isEmpty() const;

    Description& add(Description&& other, bool assignNewIds = true);

    bool hasUniqueIds() const;
    void assignNewIds();  // Preserves order of cell ids

    Description& addCreature(std::vector<ObjectDescription> const& objects, CreatureDescription const& creature, GenomeDescription const& genome = GenomeDescription());

    size_t getNumObjects() const;
    size_t getNumObjectsWithoutCreature() const;
    std::vector<ObjectDescription> getObjectsForCreature(uint64_t creatureId) const;

    DescriptionCache createCache() const;
    Description& addConnection(uint64_t const& objectId1, uint64_t const& objectId2, DescriptionCache const& cache = nullptr);
    Description& addConnection(uint64_t const& objectId1, uint64_t const& objectId2, RealVector2D const& refPosCell2, DescriptionCache const& cache = nullptr);

    ObjectDescription const& getObjectRef(uint64_t const& objectId, DescriptionCache const& cache = nullptr) const;
    ObjectDescription& getObjectRef(uint64_t const& objectId, DescriptionCache const& cache = nullptr);

    ObjectDescription& getOtherObjectRef(uint64_t id);
    ObjectDescription& getOtherObjectRef(std::set<uint64_t> const& ids);
    std::vector<ObjectDescription> getOtherObjects(std::set<uint64_t> const& ids) const;

    GenomeDescription const& getGenomeRef(uint64_t const& genomeId, DescriptionCache const& cache = nullptr) const;

    bool hasConnection(uint64_t id, uint64_t otherId) const;
    bool hasConnection(ObjectDescription const& object1, ObjectDescription const& object2) const;
    ConnectionDescription& getConnectionRef(uint64_t id, uint64_t otherId);
    ConnectionDescription const& getConnection(ObjectDescription const& object1, ObjectDescription const& object2) const;
    CreatureDescription& getCreatureRef(uint64_t id);
    CreatureDescription& getOtherCreatureRef(uint64_t id);

private:
    uint64_t getObjectIndex(uint64_t const& objectId, DescriptionCache const& cache) const;
};

struct ExtendedObjectDescription
{
    ObjectDescription object;
    std::optional<uint64_t> creatureId;
    std::optional<GenomeDescription> genome;
};
using ExtendedObjectOrEnergyDescription = std::variant<ExtendedObjectDescription, EnergyDescription>;
