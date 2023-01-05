#pragma once

#include <vector>
#include <optional>
#include <variant>
#include <cstdint>

#include "Constants.h"
#include "CellFunctionEnums.h"

struct MakeGenomeCopy
{
    auto operator<=>(MakeGenomeCopy const&) const = default;
};

struct NeuronGenomeDescription
{
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;

    NeuronGenomeDescription()
    {
        weights.resize(MAX_CHANNELS, std::vector<float>(MAX_CHANNELS, 0));
        bias.resize(MAX_CHANNELS, 0);
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
    int mode = 5;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    bool singleConstruction = false;
    bool separateConstruction = true;
    bool adaptMaxConnections = true;
    ConstructorAngleAlignment angleAlignment = ConstructorAngleAlignment_60;
    float stiffness = 1.0f;
    int constructionActivationTime = 100;

    std::variant<MakeGenomeCopy, std::vector<uint8_t>> genome = std::vector<uint8_t>();

    auto operator<=>(ConstructorGenomeDescription const&) const = default;

    ConstructorGenomeDescription& setMode(int value)
    {
        mode = value;
        return *this;
    }
    ConstructorGenomeDescription& setSingleConstruction(bool value)
    {
        singleConstruction = value;
        return *this;
    }
    ConstructorGenomeDescription& setSeparateConstruction(bool value)
    {
        separateConstruction = value;
        return *this;
    }
    ConstructorGenomeDescription& setMakeSticky(bool value)
    {
        adaptMaxConnections = value;
        return *this;
    }
    ConstructorGenomeDescription& setAngleAlignment(ConstructorAngleAlignment value)
    {
        angleAlignment = value;
        return *this;
    }
    ConstructorGenomeDescription& setStiffness(float value)
    {
        stiffness = value;
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
    ConstructorGenomeDescription& setMakeGenomeCopy()
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
    std::variant<MakeGenomeCopy, std::vector<uint8_t>> genome = std::vector<uint8_t>();

    auto operator<=>(InjectorGenomeDescription const&) const = default;

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

struct PlaceHolderGenomeDescription1
{
    auto operator<=>(PlaceHolderGenomeDescription1 const&) const = default;
};

struct PlaceHolderGenomeDescription2
{
    auto operator<=>(PlaceHolderGenomeDescription2 const&) const = default;
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
    PlaceHolderGenomeDescription1,
    PlaceHolderGenomeDescription2>>;

struct CellGenomeDescription
{
    float referenceDistance = 1.0f;
    float referenceAngle = 0;
    int color = 0;
    int maxConnections = 2;
    int executionOrderNumber = 0;

    bool inputBlocked = false;
    bool outputBlocked = false;
    CellFunctionGenomeDescription cellFunction;

    CellGenomeDescription() = default;
    auto operator<=>(CellGenomeDescription const&) const = default;

    CellGenomeDescription& setReferenceDistance(float value)
    {
        referenceDistance = value;
        return *this;
    }
    CellGenomeDescription& setReferenceAngle(float value)
    {
        referenceAngle = value;
        return *this;
    }
    CellGenomeDescription& setColor(unsigned char value)
    {
        color = value;
        return *this;
    }
    CellGenomeDescription& setMaxConnections(int value)
    {
        maxConnections = value;
        return *this;
    }
    CellGenomeDescription& setExecutionOrderNumber(int value)
    {
        executionOrderNumber = value;
        return *this;
    }
    CellGenomeDescription& setInputBlocked(bool value)
    {
        inputBlocked = value;
        return *this;
    }
    CellGenomeDescription& setOutputBlocked(bool value)
    {
        outputBlocked = value;
        return *this;
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
        if (std::holds_alternative<PlaceHolderGenomeDescription1>(*cellFunction)) {
            return CellFunction_Placeholder1;
        }
        if (std::holds_alternative<PlaceHolderGenomeDescription2>(*cellFunction)) {
            return CellFunction_Placeholder2;
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

using GenomeDescription = std::vector<CellGenomeDescription>;
