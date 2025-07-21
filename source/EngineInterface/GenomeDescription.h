#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <variant>
#include <limits>

#include "Base/Definitions.h"
#include "EngineConstants.h"
#include "CellTypeConstants.h"

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

    MEMBER(DepotGenomeDescription, EnergyDistributionMode, mode, EnergyDistributionMode_TransmittersAndConstructors);
};

struct ConstructorGenomeDescription
{
    auto operator<=>(ConstructorGenomeDescription const&) const = default;

    MEMBER(ConstructorGenomeDescription, std::optional<int>, autoTriggerInterval, 100);  // std::nullopt = manual triggering
    MEMBER(ConstructorGenomeDescription, int, geneIndex, 0);
    MEMBER(ConstructorGenomeDescription, int, constructionActivationTime, 100);
};

struct SensorGenomeDescription
{
    auto operator<=>(SensorGenomeDescription const&) const = default;

    MEMBER(SensorGenomeDescription, std::optional<int>, autoTriggerInterval, 10);  // std::nullopt = manual triggering
    MEMBER(SensorGenomeDescription, float, minDensity, 0.05f);
    MEMBER(SensorGenomeDescription, std::optional<int>, minRange, std::nullopt);
    MEMBER(SensorGenomeDescription, std::optional<int>, maxRange, std::nullopt);
    MEMBER(SensorGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(SensorGenomeDescription, SensorRestrictToCreatures, restrictToCreatures, SensorRestrictToCreatures_NoRestriction);
};

struct GeneratorGenomeDescription
{
    auto operator<=>(GeneratorGenomeDescription const&) const = default;

    MEMBER(GeneratorGenomeDescription, int, autoTriggerInterval, 100);  // 0 = no triggering, > 0 = auto trigger
    MEMBER(GeneratorGenomeDescription, GeneratorPulseType, pulseType, GeneratorPulseType_Positive);
    MEMBER(GeneratorGenomeDescription, int, alternationInterval, 1);  // Only for alternation type: 1 = alternate after each pulse, 2 = alternate after second pulse, etc.
};

struct AttackerGenomeDescription
{
    auto operator<=>(AttackerGenomeDescription const&) const = default;
};

struct InjectorGenomeDescription
{
    auto operator<=>(InjectorGenomeDescription const&) const = default;

    MEMBER(InjectorGenomeDescription, InjectorMode, mode, InjectorMode_InjectAll);
};


struct AutoBendingGenomeDescription
{
    auto operator<=>(AutoBendingGenomeDescription const&) const = default;

    MEMBER(AutoBendingGenomeDescription, float, maxAngleDeviation, 0.2f);  // Between 0 and 1
    MEMBER(AutoBendingGenomeDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1
};

struct ManualBendingGenomeDescription
{
    auto operator<=>(ManualBendingGenomeDescription const&) const = default;

    MEMBER(ManualBendingGenomeDescription, float, maxAngleDeviation, 0.2f);  // Between 0 and 1
    MEMBER(ManualBendingGenomeDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1
};

struct AngleBendingGenomeDescription
{
    auto operator<=>(AngleBendingGenomeDescription const&) const = default;

    MEMBER(AngleBendingGenomeDescription, float, maxAngleDeviation, 0.2f);  // Between 0 and 1
    MEMBER(AngleBendingGenomeDescription, float, frontBackVelRatio, 0.2f);  // Between 0 and 1
};

struct AutoCrawlingGenomeDescription
{
    auto operator<=>(AutoCrawlingGenomeDescription const&) const = default;

    MEMBER(AutoCrawlingGenomeDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(AutoCrawlingGenomeDescription, float, frontBackVelRatio, 0.2f);     // Between 0 and 1
};

struct ManualCrawlingGenomeDescription
{
    auto operator<=>(ManualCrawlingGenomeDescription const&) const = default;

    MEMBER(ManualCrawlingGenomeDescription, float, maxDistanceDeviation, 0.8f);  // Between 0 and 1
    MEMBER(ManualCrawlingGenomeDescription, float, frontBackVelRatio, 0.2f);     // Between 0 and 1
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

    MEMBER(MuscleGenomeDescription, MuscleModeGenomeDescription, mode, MuscleModeGenomeDescription());

    MuscleMode getMode() const;
};

struct DefenderGenomeDescription
{
    auto operator<=>(DefenderGenomeDescription const&) const = default;

    MEMBER(DefenderGenomeDescription, DefenderMode, mode, DefenderMode_DefendAgainstAttacker);
};

struct ReconnectorGenomeDescription
{
    auto operator<=>(ReconnectorGenomeDescription const&) const = default;

    MEMBER(ReconnectorGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(ReconnectorGenomeDescription, ReconnectorRestrictToCreatures, restrictToCreatures, ReconnectorRestrictToCreatures_NoRestriction);
};

struct DetonatorGenomeDescription
{
    auto operator<=>(DetonatorGenomeDescription const&) const = default;

    MEMBER(DetonatorGenomeDescription, int, countdown, 10);
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
    DetonatorGenomeDescription>;

struct SignalRoutingRestrictionGenomeDescription
{
    auto operator<=>(SignalRoutingRestrictionGenomeDescription const&) const = default;

    MEMBER(SignalRoutingRestrictionGenomeDescription, bool, active, false);
    MEMBER(SignalRoutingRestrictionGenomeDescription, float, baseAngle, 0.0f);
    MEMBER(SignalRoutingRestrictionGenomeDescription, float, openingAngle, 0.0f);
};

struct NodeDescription
{
    auto operator<=>(NodeDescription const&) const = default;

    MEMBER(NodeDescription, float, referenceAngle, 0.0f);
    MEMBER(NodeDescription, int, color, 0);
    MEMBER(NodeDescription, int, numAdditionalConnections, 0);

    MEMBER(NodeDescription, NeuralNetworkGenomeDescription, neuralNetwork, NeuralNetworkGenomeDescription());
    MEMBER(NodeDescription, CellTypeGenomeDescription, cellTypeData, BaseGenomeDescription());
    MEMBER(NodeDescription, SignalRoutingRestrictionGenomeDescription, signalRoutingRestriction, SignalRoutingRestrictionGenomeDescription());

    CellTypeGenome getCellType() const;
};

struct GeneDescription
{
    auto operator<=>(GeneDescription const&) const = default;

    MEMBER(GeneDescription, std::vector<NodeDescription>, nodes, {});
    MEMBER(GeneDescription, ConstructorShape, shape, ConstructorShape_Custom);
    MEMBER(GeneDescription, bool, separating, true);
    MEMBER(GeneDescription, int, numBranches, 1);   // For separating = false
    MEMBER(GeneDescription, int, numConcatenations, 1);
    MEMBER(GeneDescription, ConstructorAngleAlignment, angleAlignment, ConstructorAngleAlignment_60);
    MEMBER(GeneDescription, float, stiffness, 1.0f);
    MEMBER(GeneDescription, float, connectionDistance, 1.0f);

    static auto constexpr NumConcatenations_Infinite = std::numeric_limits<int>::max();
};

struct GenomeDescription
{
    auto operator<=>(GenomeDescription const&) const = default;

    MEMBER(GenomeDescription, std::vector<GeneDescription>, genes, {})
    MEMBER(GenomeDescription, float, frontAngle, 0.0f);
};
