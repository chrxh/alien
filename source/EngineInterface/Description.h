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

struct ConnectionDesc
{
    auto operator<=>(ConnectionDesc const&) const = default;

    MEMBER(ConnectionDesc, uint64_t, objectId, 0);
    MEMBER(ConnectionDesc, float, distance, 0.0f);
    MEMBER(ConnectionDesc, float, angleFromPrevious, 0.0f);
};

struct NeuralNetworkDesc
{
    NeuralNetworkDesc();
    auto operator<=>(NeuralNetworkDesc const&) const = default;

    MEMBER(NeuralNetworkDesc, std::vector<float>, weights, {});
    MEMBER(NeuralNetworkDesc, std::vector<float>, biases, {});
    MEMBER(NeuralNetworkDesc, std::vector<ActivationFunction>, activationFunctions, {});

    NeuralNetworkDesc& weight(int row, int col, float value);
};

struct BaseDesc
{
    auto operator<=>(BaseDesc const&) const = default;
};

struct DepotDesc
{
    auto operator<=>(DepotDesc const&) const = default;

    MEMBER(DepotDesc, float, storageLimit, 200.0f);
    MEMBER(DepotDesc, float, storedUsableEnergy, 0.0f);
};

struct ConstructorDesc
{
    auto operator<=>(ConstructorDesc const&) const = default;

    // Properties
    MEMBER(ConstructorDesc, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(ConstructorDesc, int, constructionActivationTime, 100);
    MEMBER(ConstructorDesc, float, constructionAngle, 0.0f);
    MEMBER(ConstructorDesc, ProvideEnergy, provideEnergy, ProvideEnergy_CellOnly);

    // Genome data
    MEMBER(ConstructorDesc, int, geneIndex, 0);

    // Process data
    MEMBER(ConstructorDesc, std::optional<uint64_t>, lastConstructedCellId, std::nullopt);
    MEMBER(ConstructorDesc, int, currentNodeIndex, 0);
    MEMBER(ConstructorDesc, int, currentConcatenation, 0);
    MEMBER(ConstructorDesc, int, currentBranch, 0);
};

struct TelemetryDesc
{
    auto operator<=>(TelemetryDesc const&) const = default;
};

struct DetectEnergyDesc
{
    auto operator<=>(DetectEnergyDesc const&) const = default;

    MEMBER(DetectEnergyDesc, float, minDensity, 0.05f);
};

struct DetectStructureDesc
{
    auto operator<=>(DetectStructureDesc const&) const = default;
};

struct DetectFreeCellDesc
{
    auto operator<=>(DetectFreeCellDesc const&) const = default;

    MEMBER(DetectFreeCellDesc, float, minDensity, 0.05f);
    MEMBER(DetectFreeCellDesc, std::optional<int>, restrictToColor, std::nullopt);
};

struct DetectCreatureDesc
{
    auto operator<=>(DetectCreatureDesc const&) const = default;

    MEMBER(DetectCreatureDesc, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(DetectCreatureDesc, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(DetectCreatureDesc, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(DetectCreatureDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using SensorModeDesc = std::variant<TelemetryDesc, DetectEnergyDesc, DetectStructureDesc, DetectFreeCellDesc, DetectCreatureDesc>;

struct SensorLastMatchDesc
{
    auto operator<=>(SensorLastMatchDesc const&) const = default;

    MEMBER(SensorLastMatchDesc, uint64_t, creatureId, 0);
    MEMBER(SensorLastMatchDesc, RealVector2D, pos, RealVector2D());
};

struct SensorDesc
{
    auto operator<=>(SensorDesc const&) const = default;

    MEMBER(SensorDesc, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering, value must be >= 3
    MEMBER(SensorDesc, SensorModeDesc, mode, DetectCreatureDesc());
    MEMBER(SensorDesc, int, minRange, 0);
    MEMBER(SensorDesc, int, maxRange, 255);

    // Process data
    MEMBER(SensorDesc, std::optional<SensorLastMatchDesc>, lastMatch, std::nullopt);

    SensorMode getMode() const;
};

struct GeneratorDesc
{
    auto operator<=>(GeneratorDesc const&) const = default;

    // Fixed data
    MEMBER(GeneratorDesc, int, autoTriggerInterval, 100);  // Must be >= 3
    MEMBER(GeneratorDesc, GeneratorPulseType, pulseType, GeneratorPulseType_Positive);
    MEMBER(
        GeneratorDesc,
        int,
        alternationInterval,
        20);  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.

    // Process data
    MEMBER(GeneratorDesc, int, numPulses, 0);
};

struct AttackFreeCellDesc
{
    auto operator<=>(AttackFreeCellDesc const&) const = default;

    MEMBER(AttackFreeCellDesc, std::optional<int>, restrictToColor, std::nullopt);
};

struct AttackCreatureDesc
{
    auto operator<=>(AttackCreatureDesc const&) const = default;

    MEMBER(AttackCreatureDesc, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(AttackCreatureDesc, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(AttackCreatureDesc, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(AttackCreatureDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using AttackerModeDesc = std::variant<AttackFreeCellDesc, AttackCreatureDesc>;

struct AttackerDesc
{
    auto operator<=>(AttackerDesc const&) const = default;

    MEMBER(AttackerDesc, AttackerModeDesc, mode, AttackCreatureDesc());

    AttackerMode getMode() const;
};

struct InjectorDesc
{
    InjectorDesc();
    auto operator<=>(InjectorDesc const&) const = default;

    MEMBER(InjectorDesc, int, geneIndex, 0);
};

struct AutoBendingDesc
{
    auto operator<=>(AutoBendingDesc const&) const = default;

    // Fixed data
    MEMBER(AutoBendingDesc, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(AutoBendingDesc, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AutoBendingDesc, std::optional<float>, initialAngle, std::nullopt);
    MEMBER(AutoBendingDesc, bool, forward, true);  // Current direction
    MEMBER(AutoBendingDesc, float, activation, 0);
    MEMBER(AutoBendingDesc, int, activationCountdown, 0);
    MEMBER(AutoBendingDesc, bool, impulseAlreadyApplied, false);
};

struct ManualBendingDesc
{
    auto operator<=>(ManualBendingDesc const&) const = default;

    // Fixed data
    MEMBER(ManualBendingDesc, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(ManualBendingDesc, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(ManualBendingDesc, std::optional<float>, initialAngle, std::nullopt);
    MEMBER(ManualBendingDesc, float, lastAngleDelta, 0.0f);
    MEMBER(ManualBendingDesc, bool, impulseAlreadyApplied, false);
};

struct AngleBendingDesc
{
    auto operator<=>(AngleBendingDesc const&) const = default;

    // Fixed data
    MEMBER(AngleBendingDesc, float, maxAngleDeviation, 0.2f);         // Between 0 and 1
    MEMBER(AngleBendingDesc, float, attractionRepulsionRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AngleBendingDesc, std::optional<float>, initialAngle, std::nullopt);
};

struct AutoCrawlingDesc
{
    auto operator<=>(AutoCrawlingDesc const&) const = default;

    // Fixed data
    MEMBER(AutoCrawlingDesc, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(AutoCrawlingDesc, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(AutoCrawlingDesc, std::optional<float>, initialDistance, std::nullopt);
    MEMBER(AutoCrawlingDesc, float, lastActualDistance, 0.0f);
    MEMBER(AutoCrawlingDesc, bool, forward, true);  // Current direction
    MEMBER(AutoCrawlingDesc, float, activation, 0.0f);
    MEMBER(AutoCrawlingDesc, int, activationCountdown, 0);
    MEMBER(AutoCrawlingDesc, bool, impulseAlreadyApplied, false);
};

struct ManualCrawlingDesc
{
    auto operator<=>(ManualCrawlingDesc const&) const = default;

    // Fixed data
    MEMBER(ManualCrawlingDesc, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingDesc, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1

    // Process data
    MEMBER(ManualCrawlingDesc, std::optional<float>, initialDistance, std::nullopt);
    MEMBER(ManualCrawlingDesc, float, lastActualDistance, 0.0f);
    MEMBER(ManualCrawlingDesc, float, lastDistanceDelta, 0.0f);
    MEMBER(ManualCrawlingDesc, bool, impulseAlreadyApplied, false);
};

struct DirectMovementDesc
{
    auto operator<=>(DirectMovementDesc const&) const = default;
};

using MuscleModeDesc = std::variant<
    AutoBendingDesc,
    ManualBendingDesc,
    AngleBendingDesc,
    AutoCrawlingDesc,
    ManualCrawlingDesc,
    DirectMovementDesc>;

struct MuscleDesc
{
    auto operator<=>(MuscleDesc const&) const = default;

    MEMBER(MuscleDesc, MuscleModeDesc, mode, AutoBendingDesc());

    // Additional rendering data
    MEMBER(MuscleDesc, float, lastMovementX, 0.0f);
    MEMBER(MuscleDesc, float, lastMovementY, 0.0f);

    MuscleMode getMode() const;
};

struct DefenderDesc
{
    auto operator<=>(DefenderDesc const&) const = default;

    MEMBER(DefenderDesc, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectStructureDesc
{
    auto operator<=>(ReconnectStructureDesc const&) const = default;
};

struct ReconnectFreeCellDesc
{
    auto operator<=>(ReconnectFreeCellDesc const&) const = default;

    MEMBER(ReconnectFreeCellDesc, std::optional<int>, restrictToColor, std::nullopt);
};

struct ReconnectCreatureDesc
{
    auto operator<=>(ReconnectCreatureDesc const&) const = default;

    MEMBER(ReconnectCreatureDesc, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(ReconnectCreatureDesc, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(ReconnectCreatureDesc, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectCreatureDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using ReconnectorModeDesc = std::variant<ReconnectStructureDesc, ReconnectFreeCellDesc, ReconnectCreatureDesc>;

struct ReconnectorDesc
{
    auto operator<=>(ReconnectorDesc const&) const = default;

    MEMBER(ReconnectorDesc, ReconnectorModeDesc, mode, ReconnectCreatureDesc());

    ReconnectorMode getMode() const;
};

struct DetonatorDesc
{
    auto operator<=>(DetonatorDesc const&) const = default;

    MEMBER(DetonatorDesc, DetonatorState, state, DetonatorState_Ready);
    MEMBER(DetonatorDesc, int, countdown, 60);
};

struct DigestorDesc
{
    auto operator<=>(DigestorDesc const&) const = default;

    MEMBER(DigestorDesc, float, rawEnergyConductivity, 0.5f);    // Between 0 and 1

    float getRawEnergyConversionRate() const { return 1 - _rawEnergyConductivity; }
    DigestorDesc& setRawEnergyConversionRate(float value)
    {
        _rawEnergyConductivity = 1 - value;
        return *this;
    }
};

struct SignalEntryDesc
{
    SignalEntryDesc();
    auto operator<=>(SignalEntryDesc const&) const = default;

    MEMBER(SignalEntryDesc, std::vector<float>, channels, {});
};

struct SignalDelayDesc
{
    auto operator<=>(SignalDelayDesc const&) const = default;

    MEMBER(SignalDelayDesc, int, delay, 10);
    MEMBER(SignalDelayDesc, int, numSignalEntriesInitialized, 0);
    MEMBER(SignalDelayDesc, int, ringBufferIndex, 0);
};

struct SignalRecorderDesc
{
    auto operator<=>(SignalRecorderDesc const&) const = default;

    MEMBER(SignalRecorderDesc, bool, readOnly, true);
    MEMBER(SignalRecorderDesc, SignalRecorderState, state, SignalRecorderState_Idle);
    MEMBER(SignalRecorderDesc, int, numWrittenSignalEntries, 0);
    MEMBER(SignalRecorderDesc, int, numReadSignalEntries, 0);
};

struct SignalStorageDesc
{
    auto operator<=>(SignalStorageDesc const&) const = default;

    MEMBER(SignalStorageDesc, bool, readOnly, true);
};

struct SignalIntegratorDesc
{
    auto operator<=>(SignalIntegratorDesc const&) const = default;

    MEMBER(SignalIntegratorDesc, float, newSignalWeight, 0.5f);  // Between 0 and 1
};

using MemoryModeDesc = std::variant<SignalDelayDesc, SignalRecorderDesc, SignalStorageDesc, SignalIntegratorDesc>;

struct MemoryDesc
{
    auto operator<=>(MemoryDesc const&) const = default;

    MEMBER(MemoryDesc, MemoryModeDesc, mode, SignalDelayDesc());
    MEMBER(MemoryDesc, std::vector<SignalEntryDesc>, signalEntries, {});
    MEMBER(MemoryDesc, uint8_t, channelBitMask, 0b11111111);

    MemoryMode getMode() const;
};

struct SenderDesc
{
    auto operator<=>(SenderDesc const&) const = default;

    MEMBER(SenderDesc, float, range, 15.0f);
    MEMBER(SenderDesc, int, maxTimesSent, 4);
};

struct ReceiverDesc
{
    auto operator<=>(ReceiverDesc const&) const = default;

    MEMBER(ReceiverDesc, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReceiverDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using CommunicatorModeDesc = std::variant<SenderDesc, ReceiverDesc>;

struct CommunicatorDesc
{
    auto operator<=>(CommunicatorDesc const&) const = default;

    MEMBER(CommunicatorDesc, CommunicatorModeDesc, mode, SenderDesc());

    CommunicatorMode getMode() const;
};

using CellTypeDesc = std::variant<
    BaseDesc,
    DepotDesc,
    SensorDesc,
    GeneratorDesc,
    AttackerDesc,
    InjectorDesc,
    MuscleDesc,
    DefenderDesc,
    ReconnectorDesc,
    DetonatorDesc,
    DigestorDesc,
    MemoryDesc,
    CommunicatorDesc>;

struct SignalRestrictionDesc
{
    auto operator<=>(SignalRestrictionDesc const&) const = default;

    MEMBER(SignalRestrictionDesc, SignalRestrictionMode, mode, SignalRestrictionMode_Inactive);
    MEMBER(SignalRestrictionDesc, float, baseAngle, 0);
    MEMBER(SignalRestrictionDesc, float, openingAngle, 90.0f);
};

struct SignalDesc
{
    SignalDesc();
    auto operator<=>(SignalDesc const&) const = default;

    MEMBER(SignalDesc, std::vector<float>, channels, {});
    MEMBER(SignalDesc, int, numTimesSent, 0);
};

struct StructureDesc
{
    auto operator<=>(StructureDesc const&) const = default;

    MEMBER(StructureDesc, float, energy, 100.0f);
};

struct FreeCellDesc
{
    auto operator<=>(FreeCellDesc const&) const = default;

    MEMBER(FreeCellDesc, float, energy, 100.0f);
    MEMBER(FreeCellDesc, int, age, 0);
};

struct CellDesc
{
    auto operator<=>(CellDesc const&) const = default;

    MEMBER(CellDesc, float, usableEnergy, 100.0f);
    MEMBER(CellDesc, float, rawEnergy, 0.0f);
    MEMBER(
        CellDesc,
        std::optional<float>,
        frontAngle,
        std::nullopt);  // Angle between [cell, cell->connection[0]] and front direction in reference configuration
    MEMBER(CellDesc, int, age, 0);
    MEMBER(CellDesc, CellState, cellState, CellState_Ready);

    // Creature/genome data
    MEMBER(CellDesc, std::optional<uint64_t>, creatureId, std::nullopt);
    MEMBER(CellDesc, int, nodeIndex, 0);
    MEMBER(CellDesc, int, parentNodeIndex, 0);
    MEMBER(CellDesc, int, geneIndex, 0);

    // Cell type-specific data
    MEMBER(CellDesc, NeuralNetworkDesc, neuralNetwork, NeuralNetworkDesc());
    MEMBER(CellDesc, CellTypeDesc, cellType, BaseDesc());
    MEMBER(CellDesc, std::optional<ConstructorDesc>, constructor, std::nullopt);
    MEMBER(CellDesc, SignalState, signalState, SignalState_Inactive);
    MEMBER(CellDesc, SignalDesc, signal, SignalDesc());  // For signalState == SignalState_Active
    MEMBER(CellDesc, SignalRestrictionDesc, signalRestriction, SignalRestrictionDesc());
    MEMBER(CellDesc, int, activationTime, 0);

    // Process data
    MEMBER(CellDesc, int, frontAngleId, 0);
    MEMBER(CellDesc, bool, headCell, false);

    // Additional rendering data
    MEMBER(CellDesc, CellEvent, event, CellEvent_No);
    MEMBER(CellDesc, int, eventCounter, 0);
    MEMBER(CellDesc, RealVector2D, eventPos, RealVector2D());

    CellType getCellType() const;
    CellDesc& signalAndState(std::vector<float> const& value);
    CellDesc& signalRestriction(float baseAngle, float openingAngle);
};

using ObjectTypeDesc = std::variant<StructureDesc, FreeCellDesc, CellDesc>;

struct ObjectDesc
{
    ObjectDesc(bool createIds = true);
    auto operator<=>(ObjectDesc const&) const = default;

    uint64_t _id = 0;
    ObjectDesc id(uint64_t id);
    MEMBER(ObjectDesc, std::vector<ConnectionDesc>, connections, {});
    MEMBER(ObjectDesc, RealVector2D, pos, RealVector2D());
    MEMBER(ObjectDesc, RealVector2D, vel, RealVector2D());
    MEMBER(ObjectDesc, float, stiffness, 1.0f);
    MEMBER(ObjectDesc, int, color, 0);
    MEMBER(ObjectDesc, bool, fixed, false);
    MEMBER(ObjectDesc, bool, sticky, false);
    MEMBER(ObjectDesc, ObjectTypeDesc, type, CellDesc());

    ObjectType getObjectType() const;

    StructureDesc& getStructureRef();
    StructureDesc const& getStructureRef() const;
    FreeCellDesc& getFreeCellRef();
    FreeCellDesc const& getFreeCellRef() const;
    CellDesc& getCellRef();
    CellDesc const& getCellRef() const;

    bool isConnectedTo(uint64_t id) const;
    float getAngleSpan(uint64_t connectedObjectId1, uint64_t connectedObjectId2) const;
};

struct EnergyDesc
{
    EnergyDesc();
    auto operator<=>(EnergyDesc const&) const = default;

    uint64_t _id = 0;
    EnergyDesc id(uint64_t id);
    MEMBER(EnergyDesc, RealVector2D, pos, RealVector2D());
    MEMBER(EnergyDesc, RealVector2D, vel, RealVector2D());
    MEMBER(EnergyDesc, float, energy, 0.0f);
    MEMBER(EnergyDesc, int, color, 0);
};

struct CreatureDesc
{
    CreatureDesc();
    auto operator<=>(CreatureDesc const&) const = default;

    uint64_t _id = 0;
    CreatureDesc id(uint64_t id);
    MEMBER(CreatureDesc, std::optional<uint64_t>, ancestorId, std::nullopt);
    MEMBER(CreatureDesc, int, generation, 0);
    MEMBER(CreatureDesc, int, lineageId, 0);
    MEMBER(CreatureDesc, int, numObjects, 0);
    MEMBER(CreatureDesc, uint64_t, genomeId, 0);
    MEMBER(CreatureDesc, int, haveMutationsApplied, 0);

    // Process data
    MEMBER(CreatureDesc, int, frontAngleId, 0);
};

struct _DescCache
{
    std::unordered_map<uint64_t, int> objectIdToIndex;
    std::unordered_map<uint64_t, uint64_t> creatureIdToIndex;
    std::unordered_map<uint64_t, uint64_t> genomeIdToIndex;
};
using DescCache = std::shared_ptr<_DescCache>;

struct Desc
{
    auto operator<=>(Desc const&) const = default;

    MEMBER(Desc, std::vector<ObjectDesc>, objects, {});
    MEMBER(Desc, std::vector<EnergyDesc>, energies, {});
    MEMBER(Desc, std::vector<CreatureDesc>, creatures, {});
    MEMBER(Desc, std::vector<GenomeDesc>, genomes, {});

    void clear();
    bool isEmpty() const;

    Desc& add(Desc&& other, bool assignNewIds = true);

    bool hasUniqueIds() const;
    void assignNewIds();  // Preserves order of cell ids

    Desc& addCreature(
        std::vector<ObjectDesc> const& objects,
        CreatureDesc const& creature = CreatureDesc(),
        GenomeDesc const& genome = GenomeDesc());
    Desc& addObjects(std::vector<ObjectDesc> const& objects);

    size_t getNumObjects() const;
    size_t getNumObjectsWithoutCreature() const;
    std::vector<ObjectDesc> getObjectsForCreature(uint64_t creatureId) const;

    DescCache createCache() const;
    Desc& addConnection(uint64_t const& objectId1, uint64_t const& objectId2, DescCache const& cache = nullptr);
    Desc& addConnection(uint64_t const& objectId1, uint64_t const& objectId2, RealVector2D const& refPosCell2, DescCache const& cache = nullptr);

    ObjectDesc const& getObjectRef(uint64_t const& objectId, DescCache const& cache = nullptr) const;
    ObjectDesc& getObjectRef(uint64_t const& objectId, DescCache const& cache = nullptr);

    ObjectDesc& getOtherObjectRef(uint64_t id);
    ObjectDesc& getOtherObjectRef(std::set<uint64_t> const& ids);
    std::vector<ObjectDesc> getOtherObjects(std::set<uint64_t> const& ids) const;

    GenomeDesc const& getGenomeRef(uint64_t const& genomeId, DescCache const& cache = nullptr) const;

    bool hasConnection(uint64_t id, uint64_t otherId) const;
    bool hasConnection(ObjectDesc const& object1, ObjectDesc const& object2) const;
    ConnectionDesc& getConnectionRef(uint64_t id, uint64_t otherId);
    ConnectionDesc const& getConnection(ObjectDesc const& object1, ObjectDesc const& object2) const;
    CreatureDesc& getCreatureRef(uint64_t id);
    CreatureDesc& getOtherCreatureRef(uint64_t id);

private:
    uint64_t getObjectIndex(uint64_t const& objectId, DescCache const& cache) const;
};

struct ExtendedObjectDesc
{
    ObjectDesc object;
    std::optional<uint64_t> creatureId;
    std::optional<GenomeDesc> genome;
};
using ExtendedObjectOrEnergyDesc = std::variant<ExtendedObjectDesc, EnergyDesc>;
