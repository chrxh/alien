#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <variant>
#include <limits>
#include <mdspan>

#include "Base/Definitions.h"
#include "EngineConstants.h"
#include "CellTypeConstants.h"

struct MakeGenomeCopy
{
    auto operator<=>(MakeGenomeCopy const&) const = default;
};

struct NeuralNetworkGenomeDescription
{
    std::vector<float> weights;
    std::vector<float> biases;
    std::vector<ActivationFunction> activationFunctions;

    NeuralNetworkGenomeDescription()
    {
        weights.resize(MAX_CHANNELS * MAX_CHANNELS, 0);
        biases.resize(MAX_CHANNELS, 0);
        activationFunctions.resize(MAX_CHANNELS, 0);
    }
    auto operator<=>(NeuralNetworkGenomeDescription const&) const = default;

    NeuralNetworkGenomeDescription& setWeight(int row, int col, float value)
    {
        auto md = std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        md[row, col] = value;
        return *this;
    }
    auto getWeights() const { return std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
    auto getWeights() { return std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS); }
};

struct BaseGenomeDescription
{
    auto operator<=>(BaseGenomeDescription const&) const = default;
};

struct DepotGenomeDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(DepotGenomeDescription const&) const = default;

    DepotGenomeDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct ConstructorGenomeDescription
{
    int mode = 100;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third timestep, etc.
    int constructionActivationTime = 100;

    std::variant<MakeGenomeCopy, std::vector<uint8_t>> genome = std::vector<uint8_t>();
    float constructionAngle1 = 0;
    float constructionAngle2 = 0;

    auto operator<=>(ConstructorGenomeDescription const&) const = default;

    ConstructorGenomeDescription& setMode(int value)
    {
        mode = value;
        return *this;
    }
    ConstructorGenomeDescription& setConstructionActivationTime(int value)
    {
        constructionActivationTime = value;
        return *this;
    }
    ConstructorGenomeDescription& setGenome(std::vector<uint8_t> const& value)
    {
        genome = value;
        return *this;
    }
    bool isMakeGenomeCopy() const { return std::holds_alternative<MakeGenomeCopy>(genome); }
    std::vector<uint8_t> getGenomeData() const { return std::get<std::vector<uint8_t>>(genome); }
    ConstructorGenomeDescription& setMakeSelfCopy()
    {
        genome = MakeGenomeCopy();
        return *this;
    }
};

struct SensorGenomeDescription
{
    float minDensity = 0.05f;
    std::optional<int> minRange;
    std::optional<int> maxRange;
    std::optional<int> restrictToColor;
    SensorRestrictToMutants restrictToMutants = SensorRestrictToMutants_NoRestriction;

    auto operator<=>(SensorGenomeDescription const&) const = default;

    SensorGenomeDescription& setMinDensity(float const& value)
    {
        minDensity = value;
        return *this;
    }

    SensorGenomeDescription& setColor(int value)
    {
        restrictToColor = value;
        return *this;
    }
    SensorGenomeDescription& setRestrictToMutants(SensorRestrictToMutants value)
    {
        restrictToMutants = value;
        return *this;
    }
};

struct OscillatorGenomeDescription
{
    int pulseMode = 0;        //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int alternationMode = 0;  //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    auto operator<=>(OscillatorGenomeDescription const&) const = default;

    OscillatorGenomeDescription& setPulseMode(int value)
    {
        pulseMode = value;
        return *this;
    }
    OscillatorGenomeDescription& setAlternationMode(int value)
    {
        alternationMode = value;
        return *this;
    }
};

struct AttackerGenomeDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(AttackerGenomeDescription const&) const = default;

    AttackerGenomeDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct InjectorGenomeDescription
{
    InjectorMode mode = InjectorMode_InjectAll;
    std::variant<MakeGenomeCopy, std::vector<uint8_t>> genome = std::vector<uint8_t>();

    auto operator<=>(InjectorGenomeDescription const&) const = default;

    InjectorGenomeDescription& setMode(InjectorMode value)
    {
        mode = value;
        return *this;
    }

    InjectorGenomeDescription& setGenome(std::vector<uint8_t> const& value)
    {
        genome = value;
        return *this;
    }
    bool isMakeGenomeCopy() const { return std::holds_alternative<MakeGenomeCopy>(genome); }
    std::vector<uint8_t> getGenomeData() const { return std::get<std::vector<uint8_t>>(genome); }
    InjectorGenomeDescription& setMakeSelfCopy()
    {
        genome = MakeGenomeCopy();
        return *this;
    }
};

struct MuscleGenomeDescription
{
    MuscleMode mode = MuscleMode_Movement;

    auto operator<=>(MuscleGenomeDescription const&) const = default;

    MuscleGenomeDescription& setMode(MuscleMode value)
    {
        mode = value;
        return *this;
    }
};

struct DefenderGenomeDescription
{
    DefenderMode mode = DefenderMode_DefendAgainstAttacker;

    auto operator<=>(DefenderGenomeDescription const&) const = default;

    DefenderGenomeDescription& setMode(DefenderMode value)
    {
        mode = value;
        return *this;
    }
};

struct ReconnectorGenomeDescription
{
    std::optional<int> restrictToColor;
    ReconnectorRestrictToMutants restrictToMutants = ReconnectorRestrictToMutants_NoRestriction;

    auto operator<=>(ReconnectorGenomeDescription const&) const = default;

    ReconnectorGenomeDescription& setRestrictToColor(int value)
    {
        restrictToColor = value;
        return *this;
    }
    ReconnectorGenomeDescription& setRestrictToMutants(ReconnectorRestrictToMutants value)
    {
        restrictToMutants = value;
        return *this;
    }
};

struct DetonatorGenomeDescription
{
    int countdown = 10;

    auto operator<=>(DetonatorGenomeDescription const&) const = default;

    DetonatorGenomeDescription& setCountDown(int value)
    {
        countdown = value;
        return *this;
    }
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

struct SignalRoutingRestrictionGenomeDescription
{
    bool active = false;
    float baseAngle = 0;
    float openingAngle = 0;

    auto operator<=>(SignalRoutingRestrictionGenomeDescription const&) const = default;
};

struct CellGenomeDescription
{
    float referenceAngle = 0;
    float energy = 100.0f;
    int color = 0;
    std::optional<int> numRequiredAdditionalConnections;

    NeuralNetworkGenomeDescription neuralNetwork;
    CellTypeGenomeDescription cellTypeData = BaseGenomeDescription();
    SignalRoutingRestrictionGenomeDescription signalRoutingRestriction;

    CellGenomeDescription() = default;
    auto operator<=>(CellGenomeDescription const&) const = default;

    CellGenomeDescription& setReferenceAngle(float value)
    {
        referenceAngle = value;
        return *this;
    }
    CellGenomeDescription& setEnergy(float value)
    {
        energy = value;
        return *this;
    }
    CellGenomeDescription& setColor(unsigned char value)
    {
        color = value;
        return *this;
    }
    bool hasGenome() const
    {
        auto cellType = getCellType();
        if (cellType == CellType_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(cellTypeData);
            return std::holds_alternative<std::vector<uint8_t>>(constructor.genome);
        }
        if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(cellTypeData);
            return std::holds_alternative<std::vector<uint8_t>>(injector.genome);
        }
        return false;
    }

    std::vector<uint8_t>& getGenomeRef()
    {
        auto cellType = getCellType();
        if (cellType == CellType_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(cellTypeData);
            if (std::holds_alternative<std::vector<uint8_t>>(constructor.genome)) {
                return std::get<std::vector<uint8_t>>(constructor.genome);
            }
        }
        if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(cellTypeData);
            if (std::holds_alternative<std::vector<uint8_t>>(injector.genome)) {
                return std::get<std::vector<uint8_t>>(injector.genome);
            }
        }
        THROW_NOT_IMPLEMENTED();
    }

    std::optional<std::vector<uint8_t>> getGenome() const
    {
        switch (getCellType()) {
        case CellType_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(cellTypeData);
            if (!constructor.isMakeGenomeCopy()) {
                return constructor.getGenomeData();
            } else {
                return std::nullopt;
            }
        }
        case CellType_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(cellTypeData);
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
    void setGenome(std::vector<uint8_t> const& genome)
    {
        switch (getCellType()) {
        case CellType_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(cellTypeData);
            if (!constructor.isMakeGenomeCopy()) {
                constructor.genome = genome;
            }
        } break;
        case CellType_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(cellTypeData);
            if (!injector.isMakeGenomeCopy()) {
                injector.genome = genome;
            }
        } break;
        }
    }
    std::optional<bool> isMakeGenomeCopy() const
    {
        switch (getCellType()) {
        case CellType_Constructor:
            return std::get<ConstructorGenomeDescription>(cellTypeData).isMakeGenomeCopy();
        case CellType_Injector:
            return std::get<InjectorGenomeDescription>(cellTypeData).isMakeGenomeCopy();
        default:
            return std::nullopt;
        }
    }
    CellType getCellType() const
    {
        if (std::holds_alternative<BaseGenomeDescription>(cellTypeData)) {
            return CellType_Base;
        } else if (std::holds_alternative<DepotGenomeDescription>(cellTypeData)) {
            return CellType_Depot;
        } else if (std::holds_alternative<ConstructorGenomeDescription>(cellTypeData)) {
            return CellType_Constructor;
        } else if (std::holds_alternative<SensorGenomeDescription>(cellTypeData)) {
            return CellType_Sensor;
        } else if (std::holds_alternative<OscillatorGenomeDescription>(cellTypeData)) {
            return CellType_Oscillator;
        } else if (std::holds_alternative<AttackerGenomeDescription>(cellTypeData)) {
            return CellType_Attacker;
        } else if (std::holds_alternative<InjectorGenomeDescription>(cellTypeData)) {
            return CellType_Injector;
        } else if (std::holds_alternative<MuscleGenomeDescription>(cellTypeData)) {
            return CellType_Muscle;
        } else if (std::holds_alternative<DefenderGenomeDescription>(cellTypeData)) {
            return CellType_Defender;
        } else if (std::holds_alternative<ReconnectorGenomeDescription>(cellTypeData)) {
            return CellType_Reconnector;
        } else if (std::holds_alternative<DetonatorGenomeDescription>(cellTypeData)) {
            return CellType_Detonator;
        }
        CHECK(false);
    }
    template <typename CellTypeDesc>
    CellGenomeDescription& setCellTypeData(CellTypeDesc const& value)
    {
        cellTypeData = value;
        return *this;
    }
    CellGenomeDescription& setNeuralNetwork(NeuralNetworkGenomeDescription const& value)
    {
        neuralNetwork = value;
        return *this;
    }
    CellGenomeDescription& setNumRequiredAdditionalConnections(int const& value)
    {
        numRequiredAdditionalConnections = value;
        return *this;
    }
};

struct GenomeHeaderDescription
{
    ConstructionShape shape = ConstructionShape_Custom;
    int numBranches = 1;    //between 1 and 6 in modulo
    bool separateConstruction = true;
    ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_60;
    float stiffness = 1.0f;
    float connectionDistance = 1.0f;
    int numRepetitions = 1;
    float concatenationAngle1 = 0;
    float concatenationAngle2 = 0;

    auto operator<=>(GenomeHeaderDescription const&) const = default;

    int getNumBranches() const { return separateConstruction ? 1 : (numBranches + 5) % 6 + 1; }

    GenomeHeaderDescription& setNumBranches(int value)
    {
        numBranches = value;
        return *this;
    }
    GenomeHeaderDescription& setSeparateConstruction(bool value)
    {
        separateConstruction = value;
        return *this;
    }
    GenomeHeaderDescription& setAngleAlignment(ConstructorAngleAlignment value)
    {
        angleAlignment = value;
        return *this;
    }
    GenomeHeaderDescription& setStiffness(float value)
    {
        stiffness = value;
        return *this;
    }
    GenomeHeaderDescription& setConnectionDistance(float value)
    {
        connectionDistance = value;
        return *this;
    }
    GenomeHeaderDescription& setNumRepetitions(int value)
    {
        numRepetitions = value;
        return *this;
    }
    GenomeHeaderDescription& setInfiniteRepetitions()
    {
        numRepetitions = std::numeric_limits<int>::max();
        return *this;
    }
};

struct GenomeDescription
{
    GenomeHeaderDescription header;
    std::vector<CellGenomeDescription> cells;

    auto operator<=>(GenomeDescription const&) const = default;

    GenomeDescription& setHeader(GenomeHeaderDescription const& value)
    {
        header = value;
        return *this;
    }

    GenomeDescription& setCells(std::vector<CellGenomeDescription> const& value)
    {
        cells = value;
        return *this;
    }
};
