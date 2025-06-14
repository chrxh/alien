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
    NeuralNetworkGenomeDescription()
    {
        _weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
        _biases.resize(MAX_CHANNELS, 0);
        _activationFunctions.resize(MAX_CHANNELS, ActivationFunction_Identity);
        // #TODO GCC incompatibily:
        // auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            // #TODO GCC incompatibily:
            // md[i, i] = 1.0f;
            _weights[i * MAX_CHANNELS + i] = 1.0f;
        }
    }
    auto operator<=>(NeuralNetworkGenomeDescription const&) const = default;

    NeuralNetworkGenomeDescription& weight(int row, int col, float value)
    {
        // #TODO GCC incompatibily:
        // auto md = std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        // md[row, col] = value;
        _weights[row * MAX_CHANNELS + col] = value;
        return *this;
    }
    // #TODO GCC incompatibily:
    // auto getWeights() const { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
    // auto getWeights() { return std::mdspan(_weights.data(), MAX_CHANNELS, MAX_CHANNELS); }

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

struct ConstructorGenomeDescription_New
{
    auto operator<=>(ConstructorGenomeDescription_New const&) const = default;

    MEMBER(ConstructorGenomeDescription_New, int, autoTriggerInterval, 100);    // 0 = manual (triggered by signal), > 0 = auto trigger
    MEMBER(ConstructorGenomeDescription_New, int, constructGeneIndex, 0);
    MEMBER(ConstructorGenomeDescription_New, int, constructionActivationTime, 100);
    MEMBER(ConstructorGenomeDescription_New, float, constructionAngle, 0.0f);
};

struct SensorGenomeDescription
{
    auto operator<=>(SensorGenomeDescription const&) const = default;

    MEMBER(SensorGenomeDescription, int, autoTriggerInterval, 10);  // 0 = manual (triggered by signal), > 0 = auto trigger
    MEMBER(SensorGenomeDescription, float, minDensity, 0.05f);
    MEMBER(SensorGenomeDescription, std::optional<int>, minRange, std::nullopt);
    MEMBER(SensorGenomeDescription, std::optional<int>, maxRange, std::nullopt);
    MEMBER(SensorGenomeDescription, std::optional<int>, restrictToColor, std::nullopt);
    MEMBER(SensorGenomeDescription, SensorRestrictToMutants, restrictToMutants, SensorRestrictToMutants_NoRestriction);
};

struct OscillatorGenomeDescription
{
    auto operator<=>(OscillatorGenomeDescription const&) const = default;

    MEMBER(OscillatorGenomeDescription, int, autoTriggerInterval, 100);  // 0 = no triggering, > 0 = auto trigger
    MEMBER(OscillatorGenomeDescription, int, alternationInterval, 0);  // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
};

struct AttackerGenomeDescription
{
    auto operator<=>(AttackerGenomeDescription const&) const = default;
};

struct InjectorGenomeDescription_New
{
    auto operator<=>(InjectorGenomeDescription_New const&) const = default;

    MEMBER(InjectorGenomeDescription_New, InjectorMode, mode, InjectorMode_InjectAll); // 0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.
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
    MEMBER(ReconnectorGenomeDescription, ReconnectorRestrictToMutants, restrictToMutants, ReconnectorRestrictToMutants_NoRestriction);
};

struct DetonatorGenomeDescription
{
    auto operator<=>(DetonatorGenomeDescription const&) const = default;

    MEMBER(DetonatorGenomeDescription, int, countdown, 10);
};

using CellTypeGenomeDescription_New = std::variant<
    BaseGenomeDescription,
    DepotGenomeDescription,
    ConstructorGenomeDescription_New,
    SensorGenomeDescription,
    OscillatorGenomeDescription,
    AttackerGenomeDescription,
    InjectorGenomeDescription_New,
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
    MEMBER(NodeDescription, int, numRequiredAdditionalConnections, 0);

    MEMBER(NodeDescription, NeuralNetworkGenomeDescription, neuralNetwork, NeuralNetworkGenomeDescription());
    MEMBER(NodeDescription, CellTypeGenomeDescription_New, cellTypeData, BaseGenomeDescription());
    MEMBER(NodeDescription, SignalRoutingRestrictionGenomeDescription, signalRoutingRestriction, SignalRoutingRestrictionGenomeDescription());

    CellTypeGenome getCellType() const;
};

struct GeneDescription
{
    auto operator<=>(GeneDescription const&) const = default;

    MEMBER(GeneDescription, std::vector<NodeDescription>, nodes, {});
    MEMBER(GeneDescription, ConstructionShape, shape, ConstructionShape_Custom);
    MEMBER(GeneDescription, int, numBranches, 1);   // Between 1 and 6 in modulo
    MEMBER(GeneDescription, int, numConcatenations, 1);
    MEMBER(GeneDescription, bool, separateConstruction, true);
    MEMBER(GeneDescription, ConstructorAngleAlignment, angleAlignment, ConstructorAngleAlignment_60);
    MEMBER(GeneDescription, float, stiffness, 1.0f);
    MEMBER(GeneDescription, float, connectionDistance, 1.0f);
};

struct GenomeDescription_New
{
    GenomeDescription_New();

    auto operator<=>(GenomeDescription_New const&) const = default;

    MEMBER(GenomeDescription_New, uint64_t, id, 0);
    MEMBER(GenomeDescription_New, std::vector<GeneDescription>, genes, {})
    MEMBER(GenomeDescription_New, float, frontAngle, 0.0f);
};


//*******************************************
//* OLD MODEL
//*******************************************

struct MakeGenomeCopy
{
    auto operator<=>(MakeGenomeCopy const&) const = default;
};

struct ConstructorGenomeDescription
{
    auto operator<=>(ConstructorGenomeDescription const&) const = default;

    ConstructorGenomeDescription& mode(int value)
    {
        _autoTriggerInterval = value;
        return *this;
    }
    ConstructorGenomeDescription& constructionActivationTime(int value)
    {
        _constructionActivationTime = value;
        return *this;
    }
    ConstructorGenomeDescription& genome(std::vector<uint8_t> const& value)
    {
        _genome = value;
        return *this;
    }
    bool isMakeGenomeCopy() const { return std::holds_alternative<MakeGenomeCopy>(_genome); }
    std::vector<uint8_t> getGenomeData() const { return std::get<std::vector<uint8_t>>(_genome); }
    ConstructorGenomeDescription& makeSelfCopy()
    {
        _genome = MakeGenomeCopy();
        return *this;
    }

    int _autoTriggerInterval = 100;  // 0 = manual (triggered by signal), > 0 = auto trigger
    int _constructionActivationTime = 100;

    std::variant<MakeGenomeCopy, std::vector<uint8_t>> _genome = std::vector<uint8_t>();
    float _constructionAngle1 = 0;
    float _constructionAngle2 = 0;
};

struct InjectorGenomeDescription
{
    auto operator<=>(InjectorGenomeDescription const&) const = default;

    InjectorGenomeDescription& mode(InjectorMode value)
    {
        _mode = value;
        return *this;
    }

    InjectorGenomeDescription& genome(std::vector<uint8_t> const& value)
    {
        _genome = value;
        return *this;
    }
    bool isMakeGenomeCopy() const { return std::holds_alternative<MakeGenomeCopy>(_genome); }
    std::vector<uint8_t> getGenomeData() const { return std::get<std::vector<uint8_t>>(_genome); }
    InjectorGenomeDescription& makeSelfCopy()
    {
        _genome = MakeGenomeCopy();
        return *this;
    }

    InjectorMode _mode = InjectorMode_InjectAll;
    std::variant<MakeGenomeCopy, std::vector<uint8_t>> _genome = std::vector<uint8_t>();
};

using CellTypeGenomeDescription = std::variant<
    BaseGenomeDescription,
    DepotGenomeDescription,
    ConstructorGenomeDescription,
    SensorGenomeDescription,
    OscillatorGenomeDescription,
    AttackerGenomeDescription,
    InjectorGenomeDescription,
    MuscleGenomeDescription,
    DefenderGenomeDescription,
    ReconnectorGenomeDescription,
    DetonatorGenomeDescription>;

struct CellGenomeDescription
{
    CellGenomeDescription() = default;
    auto operator<=>(CellGenomeDescription const&) const = default;

    CellGenomeDescription& referenceAngle(float value)
    {
        _referenceAngle = value;
        return *this;
    }
    CellGenomeDescription& energy(float value)
    {
        _energy = value;
        return *this;
    }
    CellGenomeDescription& color(unsigned char value)
    {
        _color = value;
        return *this;
    }
    bool hasGenome() const
    {
        auto cellType = getCellType();
        if (cellType == CellType_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(_cellTypeData);
            return std::holds_alternative<std::vector<uint8_t>>(constructor._genome);
        }
        if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(_cellTypeData);
            return std::holds_alternative<std::vector<uint8_t>>(injector._genome);
        }
        return false;
    }

    std::vector<uint8_t>& getGenomeRef()
    {
        auto cellType = getCellType();
        if (cellType == CellType_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(_cellTypeData);
            if (std::holds_alternative<std::vector<uint8_t>>(constructor._genome)) {
                return std::get<std::vector<uint8_t>>(constructor._genome);
            }
        }
        if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(_cellTypeData);
            if (std::holds_alternative<std::vector<uint8_t>>(injector._genome)) {
                return std::get<std::vector<uint8_t>>(injector._genome);
            }
        }
        THROW_NOT_IMPLEMENTED();
    }

    std::optional<std::vector<uint8_t>> getGenome() const
    {
        switch (getCellType()) {
        case CellType_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(_cellTypeData);
            if (!constructor.isMakeGenomeCopy()) {
                return constructor.getGenomeData();
            } else {
                return std::nullopt;
            }
        }
        case CellType_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(_cellTypeData);
            if (!injector.isMakeGenomeCopy()) {
                return injector.getGenomeData();
            } else {
                return std::nullopt;
            }
        }
        default:
            return std::nullopt;
        }
    }
    void genome(std::vector<uint8_t> const& genome)
    {
        switch (getCellType()) {
        case CellType_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(_cellTypeData);
            if (!constructor.isMakeGenomeCopy()) {
                constructor._genome = genome;
            }
        } break;
        case CellType_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(_cellTypeData);
            if (!injector.isMakeGenomeCopy()) {
                injector._genome = genome;
            }
        } break;
        }
    }
    std::optional<bool> isMakeGenomeCopy() const
    {
        switch (getCellType()) {
        case CellType_Constructor:
            return std::get<ConstructorGenomeDescription>(_cellTypeData).isMakeGenomeCopy();
        case CellType_Injector:
            return std::get<InjectorGenomeDescription>(_cellTypeData).isMakeGenomeCopy();
        default:
            return std::nullopt;
        }
    }
    CellType getCellType() const
    {
        if (std::holds_alternative<BaseGenomeDescription>(_cellTypeData)) {
            return CellType_Base;
        } else if (std::holds_alternative<DepotGenomeDescription>(_cellTypeData)) {
            return CellType_Depot;
        } else if (std::holds_alternative<ConstructorGenomeDescription>(_cellTypeData)) {
            return CellType_Constructor;
        } else if (std::holds_alternative<SensorGenomeDescription>(_cellTypeData)) {
            return CellType_Sensor;
        } else if (std::holds_alternative<OscillatorGenomeDescription>(_cellTypeData)) {
            return CellType_Oscillator;
        } else if (std::holds_alternative<AttackerGenomeDescription>(_cellTypeData)) {
            return CellType_Attacker;
        } else if (std::holds_alternative<InjectorGenomeDescription>(_cellTypeData)) {
            return CellType_Injector;
        } else if (std::holds_alternative<MuscleGenomeDescription>(_cellTypeData)) {
            return CellType_Muscle;
        } else if (std::holds_alternative<DefenderGenomeDescription>(_cellTypeData)) {
            return CellType_Defender;
        } else if (std::holds_alternative<ReconnectorGenomeDescription>(_cellTypeData)) {
            return CellType_Reconnector;
        } else if (std::holds_alternative<DetonatorGenomeDescription>(_cellTypeData)) {
            return CellType_Detonator;
        }
        CHECK(false);
    }
    template <typename CellTypeDesc>
    CellGenomeDescription& cellType(CellTypeDesc const& value)
    {
        _cellTypeData = value;
        return *this;
    }
    CellGenomeDescription& neuralNetwork(NeuralNetworkGenomeDescription const& value)
    {
        _neuralNetwork = value;
        return *this;
    }
    CellGenomeDescription& numRequiredAdditionalConnections(int const& value)
    {
        _numRequiredAdditionalConnections = value;
        return *this;
    }

    float _referenceAngle = 0;
    float _energy = 100.0f;
    int _color = 0;
    int _numRequiredAdditionalConnections = 0;  // Applies from second cell in genome sequence

    NeuralNetworkGenomeDescription _neuralNetwork;
    CellTypeGenomeDescription _cellTypeData = BaseGenomeDescription();
    SignalRoutingRestrictionGenomeDescription _signalRoutingRestriction;
};

struct GenomeHeaderDescription
{
    auto operator<=>(GenomeHeaderDescription const&) const = default;

    int getNumBranches() const { return _separateConstruction ? 1 : (_numBranches + 5) % 6 + 1; }

    GenomeHeaderDescription& numBranches(int value)
    {
        _numBranches = value;
        return *this;
    }
    GenomeHeaderDescription& separateConstruction(bool value)
    {
        _separateConstruction = value;
        return *this;
    }
    GenomeHeaderDescription& angleAlignment(ConstructorAngleAlignment value)
    {
        _angleAlignment = value;
        return *this;
    }
    GenomeHeaderDescription& stiffness(float value)
    {
        _stiffness = value;
        return *this;
    }
    GenomeHeaderDescription& connectionDistance(float value)
    {
        _connectionDistance = value;
        return *this;
    }
    GenomeHeaderDescription& numRepetitions(int value)
    {
        _numRepetitions = value;
        return *this;
    }
    GenomeHeaderDescription& infiniteRepetitions()
    {
        _numRepetitions = std::numeric_limits<int>::max();
        return *this;
    }
    GenomeHeaderDescription& frontAngle(float value)
    {
        _frontAngle = value;
        return *this;
    }

    ConstructionShape _shape = ConstructionShape_Custom;
    int _numBranches = 1;  //between 1 and 6 in modulo
    bool _separateConstruction = true;
    ConstructorAngleAlignment _angleAlignment = ConstructorAngleAlignment_60;
    float _stiffness = 1.0f;
    float _connectionDistance = 1.0f;
    int _numRepetitions = 1;
    float _concatenationAngle1 = 0;
    float _concatenationAngle2 = 0;
    float _frontAngle = 0;
};

struct GenomeDescription
{
    auto operator<=>(GenomeDescription const&) const = default;

    GenomeDescription& header(GenomeHeaderDescription const& value)
    {
        _header = value;
        return *this;
    }

    GenomeDescription& cells(std::vector<CellGenomeDescription> const& value)
    {
        _cells = value;
        return *this;
    }

    GenomeHeaderDescription _header;
    std::vector<CellGenomeDescription> _cells;
};
