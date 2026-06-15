#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Base/Definitions.h>

#include <EngineInterface/NeuralNetWeight.h>

#include "CellTypeConstants.h"
#include "EngineConstants.h"

struct MakeGenomeCopy;
struct BaseGenomeDesc;

struct NeuralNetGenomeDesc
{
    NeuralNetGenomeDesc();
    auto operator<=>(NeuralNetGenomeDesc const&) const = default;

    NeuralNetGenomeDesc& weight(int row, int col, NeuralNetWeight value);
    NeuralNetGenomeDesc& connectionWeight(int index, float value);
    NeuralNetGenomeDesc& bias(int index, float value);

    MEMBER(NeuralNetGenomeDesc, std::vector<NeuralNetWeight>, weights, {});
    MEMBER(NeuralNetGenomeDesc, std::vector<float>, biases, {});
    MEMBER(NeuralNetGenomeDesc, std::vector<ActivationFunction>, activationFunctions, {});
    MEMBER(NeuralNetGenomeDesc, std::vector<float>, connectionWeights, {});
};

struct BaseGenomeDesc
{
    auto operator<=>(BaseGenomeDesc const&) const = default;
};

struct DepotGenomeDesc
{
    auto operator<=>(DepotGenomeDesc const&) const = default;

    MEMBER(DepotGenomeDesc, float, storageLimit, Const::DepotStorageLimit_Default);
    MEMBER(DepotGenomeDesc, float, initialStoredUsableEnergy, Const::DepotInitialStoredUsableEnergy_Default);
};

struct ConstructorGenomeDesc
{
    auto operator<=>(ConstructorGenomeDesc const&) const = default;

    MEMBER(ConstructorGenomeDesc, std::optional<int>, autoTriggerInterval, Const::ConstructorAutoTriggerInterval_Default);  // std::nullopt = manual triggering
    MEMBER(ConstructorGenomeDesc, int, geneIndex, 0);
    MEMBER(ConstructorGenomeDesc, int, constructionActivationTime, Const::ConstructorConstructionActivationTime_Default);
    MEMBER(ConstructorGenomeDesc, float, constructionAngle, 0.0f);
    MEMBER(ConstructorGenomeDesc, ProvideEnergy, provideEnergy, ProvideEnergy_ReduceCellEnergy);
    MEMBER(ConstructorGenomeDesc, float, reservedEnergy, 0.0f);
    MEMBER(ConstructorGenomeDesc, bool, separation, false);
    MEMBER(ConstructorGenomeDesc, int, numBranches, 1);        // For separation = false
    MEMBER(ConstructorGenomeDesc, int, numConcatenations, 1);  // std::numeric_limits<int>::max() for infinite concatenations
};

struct TelemetryGenomeDesc
{
    auto operator<=>(TelemetryGenomeDesc const&) const = default;
};

struct DetectEnergyGenomeDesc
{
    auto operator<=>(DetectEnergyGenomeDesc const&) const = default;

    MEMBER(DetectEnergyGenomeDesc, float, minDensity, Const::DetectEnergyMinDensity_Default);
};

struct DetectSolidGenomeDesc
{
    auto operator<=>(DetectSolidGenomeDesc const&) const = default;
};

struct DetectFreeCellGenomeDesc
{
    auto operator<=>(DetectFreeCellGenomeDesc const&) const = default;

    MEMBER(DetectFreeCellGenomeDesc, float, minDensity, Const::DetectFreeCellMinDensity_Default);
    MEMBER(DetectFreeCellGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
};

struct DetectCreatureGenomeDesc
{
    auto operator<=>(DetectCreatureGenomeDesc const&) const = default;

    MEMBER(DetectCreatureGenomeDesc, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(DetectCreatureGenomeDesc, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(DetectCreatureGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
    MEMBER(DetectCreatureGenomeDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using SensorModeGenomeDesc =
    std::variant<TelemetryGenomeDesc, DetectEnergyGenomeDesc, DetectSolidGenomeDesc, DetectFreeCellGenomeDesc, DetectCreatureGenomeDesc>;

struct SensorGenomeDesc
{
    auto operator<=>(SensorGenomeDesc const&) const = default;

    MEMBER(SensorGenomeDesc, bool, autoTrigger, Const::SensorAutoTrigger_Default);
    MEMBER(SensorGenomeDesc, bool, tagForAttackers, Const::SensorTagForAttackers_Default);
    MEMBER(SensorGenomeDesc, SensorModeGenomeDesc, mode, DetectCreatureGenomeDesc());
    MEMBER(SensorGenomeDesc, int, minRange, Const::SensorMinRange_Default);
    MEMBER(SensorGenomeDesc, int, maxRange, Const::SensorMaxRange_Default);

    SensorMode getMode() const;
};

struct SquareSignalGenomeDesc
{
    auto operator<=>(SquareSignalGenomeDesc const&) const = default;
    MEMBER(SquareSignalGenomeDesc, int, period, Const::GeneratorPeriod_Default);
};

struct SawtoothSignalGenomeDesc
{
    auto operator<=>(SawtoothSignalGenomeDesc const&) const = default;
    MEMBER(SawtoothSignalGenomeDesc, int, period, Const::GeneratorPeriod_Default);
};

using GeneratorModeGenomeDesc = std::variant<SquareSignalGenomeDesc, SawtoothSignalGenomeDesc>;

struct GeneratorGenomeDesc
{
    auto operator<=>(GeneratorGenomeDesc const&) const = default;

    MEMBER(GeneratorGenomeDesc, bool, additive, Const::GeneratorAdditive_Default);
    MEMBER(GeneratorGenomeDesc, float, minValue, Const::GeneratorMinValue_Default);
    MEMBER(GeneratorGenomeDesc, float, maxValue, Const::GeneratorMaxValue_Default);
    MEMBER(GeneratorGenomeDesc, int, timeOffset, Const::GeneratorTimeOffset_Default);
    MEMBER(GeneratorGenomeDesc, GeneratorModeGenomeDesc, mode, SquareSignalGenomeDesc());

    GeneratorMode getMode() const;
};

struct AttackFreeCellGenomeDesc
{
    auto operator<=>(AttackFreeCellGenomeDesc const&) const = default;

    MEMBER(AttackFreeCellGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
};

struct AttackCreatureGenomeDesc
{
    auto operator<=>(AttackCreatureGenomeDesc const&) const = default;
};

using AttackerModeGenomeDesc = std::variant<AttackFreeCellGenomeDesc, AttackCreatureGenomeDesc>;

struct AttackerGenomeDesc
{
    auto operator<=>(AttackerGenomeDesc const&) const = default;

    MEMBER(AttackerGenomeDesc, AttackerModeGenomeDesc, mode, AttackCreatureGenomeDesc());

    AttackerMode getMode() const;
};

struct InjectorGenomeDesc
{
    auto operator<=>(InjectorGenomeDesc const&) const = default;

    MEMBER(InjectorGenomeDesc, int, geneIndex, Const::InjectorGeneIndex_Default);
};


struct AutoBendingGenomeDesc
{
    auto operator<=>(AutoBendingGenomeDesc const&) const = default;

    MEMBER(AutoBendingGenomeDesc, float, maxAngleDeviation, Const::MuscleMaxAngleDeviation_Default);        // Between 0 and 1
    MEMBER(AutoBendingGenomeDesc, float, forwardBackwardRatio, Const::MuscleForwardBackwardRatio_Default);  // Between 0 and 1
};

struct ManualBendingGenomeDesc
{
    auto operator<=>(ManualBendingGenomeDesc const&) const = default;

    MEMBER(ManualBendingGenomeDesc, float, maxAngleDeviation, Const::MuscleMaxAngleDeviation_Default);        // Between 0 and 1
    MEMBER(ManualBendingGenomeDesc, float, forwardBackwardRatio, Const::MuscleForwardBackwardRatio_Default);  // Between 0 and 1
};

struct AngleBendingGenomeDesc
{
    auto operator<=>(AngleBendingGenomeDesc const&) const = default;

    MEMBER(AngleBendingGenomeDesc, float, maxAngleDeviation, Const::MuscleMaxAngleDeviation_Default);              // Between 0 and 1
    MEMBER(AngleBendingGenomeDesc, float, attractionRepulsionRatio, Const::MuscleAttractionRepulsionRatio_Default);  // Between 0 and 1
};

struct AutoCrawlingGenomeDesc
{
    auto operator<=>(AutoCrawlingGenomeDesc const&) const = default;

    MEMBER(AutoCrawlingGenomeDesc, float, maxDistanceDeviation, Const::MuscleMaxDistanceDeviation_Default);  // Between 0 and 1
    MEMBER(AutoCrawlingGenomeDesc, float, forwardBackwardRatio, Const::MuscleForwardBackwardRatio_Default);  // Between 0 and 1
};

struct ManualCrawlingGenomeDesc
{
    auto operator<=>(ManualCrawlingGenomeDesc const&) const = default;

    MEMBER(ManualCrawlingGenomeDesc, float, maxDistanceDeviation, Const::MuscleMaxDistanceDeviation_Default);  // Between 0 and 1
    MEMBER(ManualCrawlingGenomeDesc, float, forwardBackwardRatio, Const::MuscleForwardBackwardRatio_Default);  // Between 0 and 1
};

struct DirectMovementGenomeDesc
{
    auto operator<=>(DirectMovementGenomeDesc const&) const = default;
};

using MuscleModeGenomeDesc = std::
    variant<AutoBendingGenomeDesc, ManualBendingGenomeDesc, AngleBendingGenomeDesc, AutoCrawlingGenomeDesc, ManualCrawlingGenomeDesc, DirectMovementGenomeDesc>;

struct MuscleGenomeDesc
{
    auto operator<=>(MuscleGenomeDesc const&) const = default;

    MEMBER(MuscleGenomeDesc, MuscleModeGenomeDesc, mode, AutoBendingGenomeDesc());

    MuscleMode getMode() const;
};

struct DefenderGenomeDesc
{
    auto operator<=>(DefenderGenomeDesc const&) const = default;

    MEMBER(DefenderGenomeDesc, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectSolidGenomeDesc
{
    auto operator<=>(ReconnectSolidGenomeDesc const&) const = default;
};

struct ReconnectFreeCellGenomeDesc
{
    auto operator<=>(ReconnectFreeCellGenomeDesc const&) const = default;

    MEMBER(ReconnectFreeCellGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
};

struct ReconnectCreatureGenomeDesc
{
    auto operator<=>(ReconnectCreatureGenomeDesc const&) const = default;

    MEMBER(ReconnectCreatureGenomeDesc, std::optional<int>, minNumCells, std::nullopt);
    MEMBER(ReconnectCreatureGenomeDesc, std::optional<int>, maxNumCells, std::nullopt);
    MEMBER(ReconnectCreatureGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
    MEMBER(ReconnectCreatureGenomeDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using ReconnectorModeGenomeDesc = std::variant<ReconnectSolidGenomeDesc, ReconnectFreeCellGenomeDesc, ReconnectCreatureGenomeDesc>;

struct ReconnectorGenomeDesc
{
    auto operator<=>(ReconnectorGenomeDesc const&) const = default;

    MEMBER(ReconnectorGenomeDesc, ReconnectorModeGenomeDesc, mode, ReconnectCreatureGenomeDesc());

    ReconnectorMode getMode() const;
};

struct DetonatorGenomeDesc
{
    auto operator<=>(DetonatorGenomeDesc const&) const = default;

    MEMBER(DetonatorGenomeDesc, int, countdown, Const::DetonatorCountdown_Default);
};

struct DigestorGenomeDesc
{
    auto operator<=>(DigestorGenomeDesc const&) const = default;

    MEMBER(DigestorGenomeDesc, float, rawEnergyConductivity, Const::DigestorRawEnergyConductivity_Default);  // Between 0 and 1

    float getRawEnergyConversionRate() const { return 1 - _rawEnergyConductivity; }
    DigestorGenomeDesc& setRawEnergyConversionRate(float value)
    {
        _rawEnergyConductivity = 1 - value;
        return *this;
    }
};

struct SignalEntryGenomeDesc
{
    SignalEntryGenomeDesc();
    auto operator<=>(SignalEntryGenomeDesc const&) const = default;

    MEMBER(SignalEntryGenomeDesc, std::vector<float>, channels, {});
};

struct SignalDelayGenomeDesc
{
    auto operator<=>(SignalDelayGenomeDesc const&) const = default;

    MEMBER(SignalDelayGenomeDesc, int, delay, Const::SignalDelay_Default);
};

struct SignalRecorderGenomeDesc
{
    auto operator<=>(SignalRecorderGenomeDesc const&) const = default;

    MEMBER(SignalRecorderGenomeDesc, bool, readOnly, true);
    MEMBER(SignalRecorderGenomeDesc, int, numWrittenSignalEntries, 0);
};

struct SignalStorageGenomeDesc
{
    auto operator<=>(SignalStorageGenomeDesc const&) const = default;

    MEMBER(SignalStorageGenomeDesc, bool, readOnly, true);
};

struct SignalIntegratorGenomeDesc
{
    auto operator<=>(SignalIntegratorGenomeDesc const&) const = default;

    MEMBER(SignalIntegratorGenomeDesc, float, newSignalWeight, Const::SignalIntegratorNewSignalWeight_Default);  // Between 0 and 1
};

using MemoryModeGenomeDesc = std::variant<SignalDelayGenomeDesc, SignalRecorderGenomeDesc, SignalStorageGenomeDesc, SignalIntegratorGenomeDesc>;

struct MemoryGenomeDesc
{
    auto operator<=>(MemoryGenomeDesc const&) const = default;

    MEMBER(MemoryGenomeDesc, MemoryModeGenomeDesc, mode, SignalDelayGenomeDesc());
    MEMBER(MemoryGenomeDesc, std::vector<SignalEntryGenomeDesc>, signalEntries, {});
    MEMBER(MemoryGenomeDesc, uint16_t, channelBitMask, 0b1111111111111111);

    MemoryMode getMode() const;
};

struct SenderGenomeDesc
{
    auto operator<=>(SenderGenomeDesc const&) const = default;

    MEMBER(SenderGenomeDesc, int, range, Const::CommunicatorRange_Default);
    MEMBER(SenderGenomeDesc, int, maxTimesSent, Const::CommunicatorMaxTimesSent_Default);
};

struct ReceiverGenomeDesc
{
    auto operator<=>(ReceiverGenomeDesc const&) const = default;

    MEMBER(ReceiverGenomeDesc, int, restrictToColors, Const::RestrictToColors_Default);
    MEMBER(ReceiverGenomeDesc, LineageRestriction, restrictToLineage, LineageRestriction_No);
};

using CommunicatorModeGenomeDesc = std::variant<SenderGenomeDesc, ReceiverGenomeDesc>;

struct CommunicatorGenomeDesc
{
    auto operator<=>(CommunicatorGenomeDesc const&) const = default;

    MEMBER(CommunicatorGenomeDesc, CommunicatorModeGenomeDesc, mode, SenderGenomeDesc());

    CommunicatorMode getMode() const;
};

struct VoidGenomeDesc
{
    auto operator<=>(VoidGenomeDesc const&) const = default;
};

using CellTypeGenomeDesc = std::variant<
    BaseGenomeDesc,
    DepotGenomeDesc,
    SensorGenomeDesc,
    GeneratorGenomeDesc,
    AttackerGenomeDesc,
    InjectorGenomeDesc,
    MuscleGenomeDesc,
    DefenderGenomeDesc,
    ReconnectorGenomeDesc,
    DetonatorGenomeDesc,
    DigestorGenomeDesc,
    MemoryGenomeDesc,
    CommunicatorGenomeDesc,
    VoidGenomeDesc>;

struct NodeDesc
{
    auto operator<=>(NodeDesc const&) const = default;

    MEMBER(NodeDesc, float, referenceAngle, 0.0f);
    MEMBER(NodeDesc, int, color, 0);

    MEMBER(NodeDesc, NeuralNetGenomeDesc, neuralNetwork, NeuralNetGenomeDesc());
    MEMBER(NodeDesc, CellTypeGenomeDesc, cellType, BaseGenomeDesc());
    MEMBER(NodeDesc, std::optional<ConstructorGenomeDesc>, constructor, std::nullopt);

    CellType getCellType() const;
};

struct GeneDesc
{
    auto operator<=>(GeneDesc const&) const = default;

    MEMBER(GeneDesc, std::string, name, "");
    MEMBER(GeneDesc, std::vector<NodeDesc>, nodes, {});
    MEMBER(GeneDesc, ConstructorShape, shape, ConstructorShape_Segment);
    MEMBER(GeneDesc, float, stiffness, 1.0f);
    MEMBER(GeneDesc, float, connectionDistance, 1.0f);
    MEMBER(GeneDesc, bool, homogeneCellType, false);
};

struct NeuronMutationDesc
{
    auto operator<=>(NeuronMutationDesc const&) const = default;

    MEMBER(NeuronMutationDesc, float, nodeProbability, 0.0f);
    MEMBER(NeuronMutationDesc, float, weightSigma, 0.0f);
    MEMBER(NeuronMutationDesc, float, biasSigma, 0.0f);
    MEMBER(NeuronMutationDesc, float, activationFunctionProbability, 0.0f);
};

struct ConnectionMutationDesc
{
    auto operator<=>(ConnectionMutationDesc const&) const = default;

    MEMBER(ConnectionMutationDesc, float, nodeProbability, 0.0f);
    MEMBER(ConnectionMutationDesc, float, sigma, 0.0f);
};

struct CellTypePropertiesMutationDesc
{
    auto operator<=>(CellTypePropertiesMutationDesc const&) const = default;

    MEMBER(CellTypePropertiesMutationDesc, float, nodeProbability, 0.0f);
    MEMBER(CellTypePropertiesMutationDesc, float, sigma, 0.0f);
    MEMBER(CellTypePropertiesMutationDesc, float, discreteChangeProbability, 0.0f);
};

struct CellTypeModeMutationDesc
{
    auto operator<=>(CellTypeModeMutationDesc const&) const = default;

    MEMBER(CellTypeModeMutationDesc, float, nodeProbability, 0.0f);
};

struct CellTypeMutationDesc
{
    auto operator<=>(CellTypeMutationDesc const&) const = default;

    MEMBER(CellTypeMutationDesc, float, nodeProbability, 0.0f);
};

struct VoidMutationDesc
{
    auto operator<=>(VoidMutationDesc const&) const = default;

    MEMBER(VoidMutationDesc, float, nodeProbability, 0.0f);
};

struct ConstructorMutationDesc
{
    auto operator<=>(ConstructorMutationDesc const&) const = default;

    MEMBER(ConstructorMutationDesc, float, nodeProbability, 0.0f);
    MEMBER(ConstructorMutationDesc, float, sigma, 0.0f);
    MEMBER(ConstructorMutationDesc, float, discreteChangeProbability, 0.0f);
    MEMBER(ConstructorMutationDesc, float, existConstructorProbability, 0.0f);
};

struct MutationRatesDesc
{
    using NeuronMutationArray = std::array<NeuronMutationDesc, 2>;
    using ConnectionMutationArray = std::array<ConnectionMutationDesc, 2>;
    using CellTypePropertiesMutationArray = std::array<CellTypePropertiesMutationDesc, 2>;
    using ConstructorMutationArray = std::array<ConstructorMutationDesc, 2>;

    auto operator<=>(MutationRatesDesc const&) const = default;

    MEMBER(MutationRatesDesc, NeuronMutationArray, neuronMutations, NeuronMutationArray());
    MEMBER(MutationRatesDesc, ConnectionMutationArray, connectionMutations, ConnectionMutationArray());
    MEMBER(MutationRatesDesc, CellTypePropertiesMutationArray, cellTypePropertiesMutations, CellTypePropertiesMutationArray());
    MEMBER(MutationRatesDesc, CellTypeModeMutationDesc, cellTypeModeMutation, CellTypeModeMutationDesc());
    MEMBER(MutationRatesDesc, CellTypeMutationDesc, cellTypeMutation, CellTypeMutationDesc());
    MEMBER(MutationRatesDesc, VoidMutationDesc, voidMutation, VoidMutationDesc());
    MEMBER(MutationRatesDesc, ConstructorMutationArray, constructorMutations, ConstructorMutationArray());

    std::vector<std::string> getActiveMutationTypes() const;
};

struct GenomeDesc
{
    GenomeDesc();
    auto operator<=>(GenomeDesc const&) const = default;
    bool equalWithoutId(GenomeDesc const& other) const;

    uint64_t _id = 0;
    GenomeDesc id(uint64_t id);
    MEMBER(GenomeDesc, std::string, name, "");
    MEMBER(GenomeDesc, std::vector<GeneDesc>, genes, {})
    MEMBER(GenomeDesc, int, lineageId, 0);
    MEMBER(GenomeDesc, std::optional<int>, prevLineageId, std::nullopt);
    MEMBER(GenomeDesc, float, frontAngle, 0.0f);
    MEMBER(GenomeDesc, float, accumulatedMutations, 0.0f);
    MEMBER(GenomeDesc, bool, resistanceToInjection, false);

    MEMBER(GenomeDesc, MutationRatesDesc, mutationRates, MutationRatesDesc());
};

struct SubGenomeDesc
{
    GenomeDesc genome;
    int startIndex = 0;
    bool trimmed = false;

    bool operator==(const SubGenomeDesc&) const = default;
};

// Include hash specializations
#include "GenomeDescHash.h"
