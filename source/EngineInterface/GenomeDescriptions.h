#pragma once

#include <cstdint>
#include <vector>
#include <optional>
#include <variant>

#include "Base/Definitions.h"
#include "FundamentalConstants.h"
#include "CellFunctionConstants.h"

struct MakeGenomeCopy
{
    auto operator<=>(MakeGenomeCopy const&) const = default;
};

struct NeuronGenomeDescription
{
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    NeuronGenomeDescription()
    {
        weights.resize(MAX_CHANNELS, std::vector<float>(MAX_CHANNELS, 0));
        biases.resize(MAX_CHANNELS, 0);
    }
    auto operator<=>(NeuronGenomeDescription const&) const = default;
};

struct TransmitterGenomeDescription
{
    EnergyDistributionMode mode = EnergyDistributionMode_TransmittersAndConstructors;

    auto operator<=>(TransmitterGenomeDescription const&) const = default;

    TransmitterGenomeDescription& setMode(EnergyDistributionMode value)
    {
        mode = value;
        return *this;
    }
};

struct ConstructorGenomeDescription
{
    int mode = 13;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
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
    std::optional<float> fixedAngle;   //nullopt = entire neighborhood
    float minDensity = 0.3f;
    int color = 0;

    auto operator<=>(SensorGenomeDescription const&) const = default;

    SensorMode getSensorMode() const { return fixedAngle.has_value() ? SensorMode_FixedAngle : SensorMode_Neighborhood; }

    SensorGenomeDescription& setFixedAngle(float const& value)
    {
        fixedAngle = value;
        return *this;
    }

    SensorGenomeDescription& setMinDensity(float const& value)
    {
        minDensity = value;
        return *this;
    }

    SensorGenomeDescription& setColor(int value)
    {
        color = value;
        return *this;
    }
};

struct NerveGenomeDescription
{
    int pulseMode = 0;        //0 = none, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    int alternationMode = 0;  //0 = none, 1 = alternate after each pulse, 2 = alternate after second pulse, 3 = alternate after third pulse, etc.

    auto operator<=>(NerveGenomeDescription const&) const = default;

    NerveGenomeDescription& setPulseMode(int value)
    {
        pulseMode = value;
        return *this;
    }
    NerveGenomeDescription& setAlternationMode(int value)
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
    InjectorGenomeDescription& setMakeGenomeCopy()
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

struct PlaceHolderGenomeDescription
{
    auto operator<=>(PlaceHolderGenomeDescription const&) const = default;
};

using CellFunctionGenomeDescription = std::optional<std::variant<
    NeuronGenomeDescription,
    TransmitterGenomeDescription,
    ConstructorGenomeDescription,
    SensorGenomeDescription,
    NerveGenomeDescription,
    AttackerGenomeDescription,
    InjectorGenomeDescription,
    MuscleGenomeDescription,
    DefenderGenomeDescription,
    PlaceHolderGenomeDescription>>;

struct CellGenomeDescription
{
    float referenceAngle = 0;
    float energy = 100.0f;
    int color = 0;
    std::optional<int> numRequiredAdditionalConnections;
    int executionOrderNumber = 0;

    std::optional<int> inputExecutionOrderNumber;
    bool outputBlocked = false;
    CellFunctionGenomeDescription cellFunction;

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
    CellGenomeDescription& setExecutionOrderNumber(int value)
    {
        executionOrderNumber = value;
        return *this;
    }
    CellGenomeDescription& setInputExecutionOrderNumber(int value)
    {
        inputExecutionOrderNumber = value;
        return *this;
    }
    CellGenomeDescription& setOutputBlocked(bool value)
    {
        outputBlocked = value;
        return *this;
    }
    bool hasGenome() const
    {
        auto cellFunctionType = getCellFunctionType();
        if (cellFunctionType == CellFunction_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cellFunction);
            return std::holds_alternative<std::vector<uint8_t>>(constructor.genome);
        }
        if (cellFunctionType == CellFunction_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(*cellFunction);
            return std::holds_alternative<std::vector<uint8_t>>(injector.genome);
        }
        return false;
    }

    std::vector<uint8_t>& getGenomeRef()
    {
        auto cellFunctionType = getCellFunctionType();
        if (cellFunctionType == CellFunction_Constructor) {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cellFunction);
            if (std::holds_alternative<std::vector<uint8_t>>(constructor.genome)) {
                return std::get<std::vector<uint8_t>>(constructor.genome);
            }
        }
        if (cellFunctionType == CellFunction_Injector) {
            auto& injector = std::get<InjectorGenomeDescription>(*cellFunction);
            if (std::holds_alternative<std::vector<uint8_t>>(injector.genome)) {
                return std::get<std::vector<uint8_t>>(injector.genome);
            }
        }
        THROW_NOT_IMPLEMENTED();
    }

    std::optional<std::vector<uint8_t>> getGenome() const
    {
        switch (getCellFunctionType()) {
        case CellFunction_Constructor: {
            auto const& constructor = std::get<ConstructorGenomeDescription>(*cellFunction);
            if (!constructor.isMakeGenomeCopy()) {
                return constructor.getGenomeData();
            } else {
                return std::nullopt;
            }
        }
        case CellFunction_Injector: {
            auto const& injector = std::get<InjectorGenomeDescription>(*cellFunction);
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
        switch (getCellFunctionType()) {
        case CellFunction_Constructor: {
            auto& constructor = std::get<ConstructorGenomeDescription>(*cellFunction);
            if (!constructor.isMakeGenomeCopy()) {
                constructor.genome = genome;
            }
        } break;
        case CellFunction_Injector: {
            auto& injector = std::get<InjectorGenomeDescription>(*cellFunction);
            if (!injector.isMakeGenomeCopy()) {
                injector.genome = genome;
            }
        } break;
        }
    }
    std::optional<bool> isMakeGenomeCopy() const
    {
        switch (getCellFunctionType()) {
        case CellFunction_Constructor:
            return std::get<ConstructorGenomeDescription>(*cellFunction).isMakeGenomeCopy();
        case CellFunction_Injector:
            return std::get<InjectorGenomeDescription>(*cellFunction).isMakeGenomeCopy();
        default:
            return std::nullopt;
        }
    }
    CellFunction getCellFunctionType() const
    {
        if (!cellFunction) {
            return CellFunction_None;
        }
        if (std::holds_alternative<NeuronGenomeDescription>(*cellFunction)) {
            return CellFunction_Neuron;
        }
        if (std::holds_alternative<TransmitterGenomeDescription>(*cellFunction)) {
            return CellFunction_Transmitter;
        }
        if (std::holds_alternative<ConstructorGenomeDescription>(*cellFunction)) {
            return CellFunction_Constructor;
        }
        if (std::holds_alternative<SensorGenomeDescription>(*cellFunction)) {
            return CellFunction_Sensor;
        }
        if (std::holds_alternative<NerveGenomeDescription>(*cellFunction)) {
            return CellFunction_Nerve;
        }
        if (std::holds_alternative<AttackerGenomeDescription>(*cellFunction)) {
            return CellFunction_Attacker;
        }
        if (std::holds_alternative<InjectorGenomeDescription>(*cellFunction)) {
            return CellFunction_Injector;
        }
        if (std::holds_alternative<MuscleGenomeDescription>(*cellFunction)) {
            return CellFunction_Muscle;
        }
        if (std::holds_alternative<DefenderGenomeDescription>(*cellFunction)) {
            return CellFunction_Defender;
        }
        if (std::holds_alternative<PlaceHolderGenomeDescription>(*cellFunction)) {
            return CellFunction_Placeholder;
        }
        return CellFunction_None;
    }
    template <typename CellFunctionDesc>
    CellGenomeDescription& setCellFunction(CellFunctionDesc const& value)
    {
        cellFunction = value;
        return *this;
    }
};

struct GenomeHeaderDescription
{
    ConstructionShape shape = ConstructionShape_Custom;
    bool singleConstruction = false;
    bool separateConstruction = true;
    ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_60;
    float stiffness = 1.0f;
    float connectionDistance = 1.0f;
    int numRepetitions = 1;
    float concatenationAngle1 = 0;
    float concatenationAngle2 = 0;

    auto operator<=>(GenomeHeaderDescription const&) const = default;

    GenomeHeaderDescription& setSingleConstruction(bool value)
    {
        singleConstruction = value;
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
