#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Base/Definitions.h>

#include "CellTypeConstants.h"
#include "EngineConstants.h"

struct MakeGenomeCopy;
struct BaseGenomeDescription;

struct NeuralNetworkGenomeDescription
{
    NeuralNetworkGenomeDescription();
    auto operator<=>(NeuralNetworkGenomeDescription const&) const = default;

    NeuralNetworkGenomeDescription& weight(int row, int col, float value);

    MEMBER(NeuralNetworkGenomeDescription, std::vector<float>, weights, {});
    MEMBER(NeuralNetworkGenomeDescription, std::vector<float>, biases, {});
    MEMBER(NeuralNetworkGenomeDescription, std::vector<ActivationFunction>, activationFunctions, {});
};

struct BaseGenomeDescription
{
    auto operator<=>(BaseGenomeDescription const&) const = default;
};

struct DepotGenomeDescription
{
    auto operator<=>(DepotGenomeDescription const&) const = default;

    MEMBER(DepotGenomeDescription, float, storageLimit, 200.0f);
    MEMBER(DepotGenomeDescription, float, initialStoredUsableEnergy, 0.0f);
};

struct ConstructorGenomeDescription
{
    auto operator<=>(ConstructorGenomeDescription const&) const = default;

    MEMBER(ConstructorGenomeDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering
    MEMBER(ConstructorGenomeDescription, int, geneIndex, 0);
    MEMBER(ConstructorGenomeDescription, int, constructionActivationTime, 100);
    MEMBER(ConstructorGenomeDescription, float, constructionAngle, 0.0f);
    MEMBER(ConstructorGenomeDescription, ProvideEnergy, provideEnergy, ProvideEnergy_CellOnly);
};

struct TelemetryGenomeDescription
{
    auto operator<=>(TelemetryGenomeDescription const&) const = default;
};

struct DetectEnergyGenomeDescription
{
    auto operator<=>(DetectEnergyGenomeDescription const&) const = default;

    MEMBER(DetectEnergyGenomeDescription, float, minDensity, 1.0f);
};

struct DetectStructureGenomeDescription
{
    auto operator<=>(DetectStructureGenomeDescription const&) const = default;
};

struct DetectFreeCellGenomeDescription
{
    auto operator<=>(DetectFreeCellGenomeDescription const&) const = default;

    MEMBER(DetectFreeCellGenomeDescription, float, minDensity, 0.5f);
    MEMBER(DetectFreeCellGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct DetectCreatureGenomeDescription
{
    auto operator<=>(DetectCreatureGenomeDescription const&) const = default;

    MEMBER(DetectCreatureGenomeDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(DetectCreatureGenomeDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(DetectCreatureGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(DetectCreatureGenomeDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using SensorModeGenomeDescription = std::variant<
    TelemetryGenomeDescription,
    DetectEnergyGenomeDescription,
    DetectStructureGenomeDescription,
    DetectFreeCellGenomeDescription,
    DetectCreatureGenomeDescription>;

struct SensorGenomeDescription
{
    auto operator<=>(SensorGenomeDescription const&) const = default;

    MEMBER(SensorGenomeDescription, std::optional<int>, autoTriggerInterval, 10);  // std::nullopt = manual triggering
    MEMBER(SensorGenomeDescription, SensorModeGenomeDescription, mode, DetectCreatureGenomeDescription());
    MEMBER(SensorGenomeDescription, int, minRange, 0);
    MEMBER(SensorGenomeDescription, int, maxRange, 255);

    SensorMode getMode() const;
};

struct GeneratorGenomeDescription
{
    auto operator<=>(GeneratorGenomeDescription const&) const = default;

    MEMBER(GeneratorGenomeDescription, int, autoTriggerInterval, 100);  // 0 = no triggering, > 0 = auto trigger
    MEMBER(GeneratorGenomeDescription, GeneratorPulseType, pulseType, GeneratorPulseType_Positive);
    MEMBER(
        GeneratorGenomeDescription,
        int,
        alternationInterval,
        20);  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
};

struct AttackFreeCellGenomeDescription
{
    auto operator<=>(AttackFreeCellGenomeDescription const&) const = default;

    MEMBER(AttackFreeCellGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct AttackCreatureGenomeDescription
{
    auto operator<=>(AttackCreatureGenomeDescription const&) const = default;

    MEMBER(AttackCreatureGenomeDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(AttackCreatureGenomeDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(AttackCreatureGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(AttackCreatureGenomeDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using AttackerModeGenomeDescription = std::variant<AttackFreeCellGenomeDescription, AttackCreatureGenomeDescription>;

struct AttackerGenomeDescription
{
    auto operator<=>(AttackerGenomeDescription const&) const = default;

    MEMBER(AttackerGenomeDescription, AttackerModeGenomeDescription, mode, AttackCreatureGenomeDescription());

    AttackerMode getMode() const;
};

struct InjectorGenomeDescription
{
    auto operator<=>(InjectorGenomeDescription const&) const = default;

    MEMBER(InjectorGenomeDescription, int, geneIndex, 0);
};


struct AutoBendingGenomeDescription
{
    auto operator<=>(AutoBendingGenomeDescription const&) const = default;

    MEMBER(AutoBendingGenomeDescription, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(AutoBendingGenomeDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1
};

struct ManualBendingGenomeDescription
{
    auto operator<=>(ManualBendingGenomeDescription const&) const = default;

    MEMBER(ManualBendingGenomeDescription, float, maxAngleDeviation, 0.2f);     // Between 0 and 1
    MEMBER(ManualBendingGenomeDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1
};

struct AngleBendingGenomeDescription
{
    auto operator<=>(AngleBendingGenomeDescription const&) const = default;

    MEMBER(AngleBendingGenomeDescription, float, maxAngleDeviation, 0.2f);         // Between 0 and 1
    MEMBER(AngleBendingGenomeDescription, float, attractionRepulsionRatio, 0.8f);  // Between 0 and 1
};

struct AutoCrawlingGenomeDescription
{
    auto operator<=>(AutoCrawlingGenomeDescription const&) const = default;

    MEMBER(AutoCrawlingGenomeDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(AutoCrawlingGenomeDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1
};

struct ManualCrawlingGenomeDescription
{
    auto operator<=>(ManualCrawlingGenomeDescription const&) const = default;

    MEMBER(ManualCrawlingGenomeDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingGenomeDescription, float, forwardBackwardRatio, 0.8f);  // Between 0 and 1
};

struct DirectMovementGenomeDescription
{
    auto operator<=>(DirectMovementGenomeDescription const&) const = default;
};

using MuscleModeGenomeDescription = std::variant<
    AutoBendingGenomeDescription,
    ManualBendingGenomeDescription,
    AngleBendingGenomeDescription,
    AutoCrawlingGenomeDescription,
    ManualCrawlingGenomeDescription,
    DirectMovementGenomeDescription>;

struct MuscleGenomeDescription
{
    auto operator<=>(MuscleGenomeDescription const&) const = default;

    MEMBER(MuscleGenomeDescription, MuscleModeGenomeDescription, mode, AutoBendingGenomeDescription());

    MuscleMode getMode() const;
};

struct DefenderGenomeDescription
{
    auto operator<=>(DefenderGenomeDescription const&) const = default;

    MEMBER(DefenderGenomeDescription, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectStructureGenomeDescription
{
    auto operator<=>(ReconnectStructureGenomeDescription const&) const = default;
};

struct ReconnectFreeCellGenomeDescription
{
    auto operator<=>(ReconnectFreeCellGenomeDescription const&) const = default;

    MEMBER(ReconnectFreeCellGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
};

struct ReconnectCreatureGenomeDescription
{
    auto operator<=>(ReconnectCreatureGenomeDescription const&) const = default;

    MEMBER(ReconnectCreatureGenomeDescription, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(ReconnectCreatureGenomeDescription, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(ReconnectCreatureGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectCreatureGenomeDescription, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using ReconnectorModeGenomeDescription = std::variant<ReconnectStructureGenomeDescription, ReconnectFreeCellGenomeDescription, ReconnectCreatureGenomeDescription>;

struct ReconnectorGenomeDescription
{
    auto operator<=>(ReconnectorGenomeDescription const&) const = default;

    MEMBER(ReconnectorGenomeDescription, ReconnectorModeGenomeDescription, mode, ReconnectCreatureGenomeDescription());

    ReconnectorMode getMode() const;
};

struct DetonatorGenomeDescription
{
    auto operator<=>(DetonatorGenomeDescription const&) const = default;

    MEMBER(DetonatorGenomeDescription, int, countdown, 10);
};

struct DigestorGenomeDescription
{
    auto operator<=>(DigestorGenomeDescription const&) const = default;

    MEMBER(DigestorGenomeDescription, float, rawEnergyConductivity, 0.5f);  // Between 0 and 1

    float getRawEnergyConversionRate() const
    {
        return 1 - _rawEnergyConductivity;
    }
    DigestorGenomeDescription& setRawEnergyConversionRate(float value)
    {
        _rawEnergyConductivity = 1 - value;
        return *this;
    }
};

struct SignalEntryGenomeDescription
{
    SignalEntryGenomeDescription();
    auto operator<=>(SignalEntryGenomeDescription const&) const = default;

    MEMBER(SignalEntryGenomeDescription, std::vector<float>, channels, {});
};

struct SignalDelayGenomeDescription
{
    auto operator<=>(SignalDelayGenomeDescription const&) const = default;

    MEMBER(SignalDelayGenomeDescription, int, delay, 10);
};

struct SignalRecorderGenomeDescription
{
    auto operator<=>(SignalRecorderGenomeDescription const&) const = default;

    MEMBER(SignalRecorderGenomeDescription, bool, readOnly, true);
    MEMBER(SignalRecorderGenomeDescription, int, numSavedSignalEntries, 0);
};

struct SignalStorageGenomeDescription
{
    auto operator<=>(SignalStorageGenomeDescription const&) const = default;
};

struct SignalIntegratorGenomeDescription
{
    auto operator<=>(SignalIntegratorGenomeDescription const&) const = default;

    MEMBER(SignalIntegratorGenomeDescription, float, newSignalWeight, 1.0f);  // Between 0 and 1
};

using MemoryModeGenomeDescription = std::variant<SignalDelayGenomeDescription, SignalRecorderGenomeDescription, SignalStorageGenomeDescription, SignalIntegratorGenomeDescription>;

struct MemoryGenomeDescription
{
    auto operator<=>(MemoryGenomeDescription const&) const = default;

    MEMBER(MemoryGenomeDescription, MemoryModeGenomeDescription, mode, SignalDelayGenomeDescription());
    MEMBER(MemoryGenomeDescription, std::vector<SignalEntryGenomeDescription>, signalEntries, {});

    MemoryMode getMode() const;
};

using CellTypeGenomeDescription = std::variant<
    BaseGenomeDescription,
    DepotGenomeDescription,
    ConstructorGenomeDescription,
    SensorGenomeDescription,
    GeneratorGenomeDescription,
    AttackerGenomeDescription,
    InjectorGenomeDescription,
    MuscleGenomeDescription,
    DefenderGenomeDescription,
    ReconnectorGenomeDescription,
    DetonatorGenomeDescription,
    DigestorGenomeDescription,
    MemoryGenomeDescription>;

struct SignalRestrictionGenomeDescription
{
    auto operator<=>(SignalRestrictionGenomeDescription const&) const = default;

    MEMBER(SignalRestrictionGenomeDescription, bool, active, false);
    MEMBER(SignalRestrictionGenomeDescription, float, baseAngle, 0.0f);
    MEMBER(SignalRestrictionGenomeDescription, float, openingAngle, 90.0f);
};

struct NodeDescription
{
    auto operator<=>(NodeDescription const&) const = default;

    MEMBER(NodeDescription, float, referenceAngle, 0.0f);
    MEMBER(NodeDescription, int, color, 0);
    MEMBER(NodeDescription, int, numAdditionalConnections, 0);

    MEMBER(NodeDescription, NeuralNetworkGenomeDescription, neuralNetwork, NeuralNetworkGenomeDescription());
    MEMBER(NodeDescription, CellTypeGenomeDescription, cellType, BaseGenomeDescription());
    MEMBER(NodeDescription, SignalRestrictionGenomeDescription, signalRestriction, SignalRestrictionGenomeDescription());

    CellTypeGenome getCellType() const;
};

struct GeneDescription
{
    auto operator<=>(GeneDescription const&) const = default;

    MEMBER(GeneDescription, std::string, name, "");
    MEMBER(GeneDescription, std::vector<NodeDescription>, nodes, {});
    MEMBER(GeneDescription, ConstructorShape, shape, ConstructorShape_Custom);
    MEMBER(GeneDescription, bool, separation, false);
    MEMBER(GeneDescription, int, numBranches, 1);        // For separation = false
    MEMBER(GeneDescription, int, numConcatenations, 1);  // std::numeric_limits<int>::max() for infinite concatenations
    MEMBER(GeneDescription, ConstructorAngleAlignment, angleAlignment, ConstructorAngleAlignment_60);
    MEMBER(GeneDescription, float, stiffness, 1.0f);
    MEMBER(GeneDescription, float, connectionDistance, 1.0f);

    static auto constexpr NumConcatenations_Infinite = std::numeric_limits<int>::max();
};

struct GenomeDescription
{
    GenomeDescription();
    auto operator<=>(GenomeDescription const&) const = default;

    uint64_t _id = 0;
    GenomeDescription id(uint64_t id);
    MEMBER(GenomeDescription, std::string, name, "");
    MEMBER(GenomeDescription, std::vector<GeneDescription>, genes, {})
    MEMBER(GenomeDescription, float, frontAngle, 0.0f);
};

struct SubGenomeDescription
{
    GenomeDescription genome;
    int startIndex = 0;
    bool trimmed = false;

    bool operator==(const SubGenomeDescription&) const = default;
};

// Include hash specializations
#include "GenomeDescriptionHash.h"
