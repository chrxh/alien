#pragma once

#include <vector>
#include <optional>
#include <variant>

#include "Constants.h"
#include "Enums.h"

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
    auto operator<=>(TransmitterGenomeDescription const&) const = default;
};

struct ConstructorGenomeDescription
{
    int mode = 1;
    bool singleConstruction = false;
    bool separateConstruction = true;
    bool makeSticky = false;
    int angleAlignment = 0;
    std::vector<uint8_t> genome;

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
        makeSticky = value;
        return *this;
    }
    ConstructorGenomeDescription& setAngleAlignment(int value)
    {
        angleAlignment = value;
        return *this;
    }
    ConstructorGenomeDescription& setGenome(std::vector<uint8_t> const& value)
    {
        genome = value;
        return *this;
    }
};

struct SensorGenomeDescription
{
    Enums::SensorMode mode = Enums::SensorMode_AllAngles;
    int color = 0;

    auto operator<=>(SensorGenomeDescription const&) const = default;
};

struct NerveGenomeDescription
{
    auto operator<=>(NerveGenomeDescription const&) const = default;
};

struct AttackerGenomeDescription
{
    auto operator<=>(AttackerGenomeDescription const&) const = default;
};

struct InjectorGenomeDescription
{
    std::vector<uint8_t> genome;

    auto operator<=>(InjectorGenomeDescription const&) const = default;
};

struct MuscleGenomeDescription
{
    auto operator<=>(MuscleGenomeDescription const&) const = default;
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
    uint64_t id = 0;

    float referenceDistance = 1.0f;
    float referenceAngle = 0.0f;
    int color = 0;
    int maxConnections = 0;
    int executionOrderNumber = 0;

    bool inputBlocked = false;
    bool outputBlocked = false;
    CellFunctionGenomeDescription cellFunction;

    CellGenomeDescription() = default;
    auto operator<=>(CellGenomeDescription const&) const = default;

    CellGenomeDescription& setId(uint64_t value)
    {
        id = value;
        return *this;
    }
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
    Enums::CellFunction getCellFunctionType() const
    {
        if (!cellFunction) {
            return Enums::CellFunction_None;
        }
        if (std::holds_alternative<NeuronGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Neuron;
        }
        if (std::holds_alternative<TransmitterGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Transmitter;
        }
        if (std::holds_alternative<ConstructorGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Constructor;
        }
        if (std::holds_alternative<SensorGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Sensor;
        }
        if (std::holds_alternative<NerveGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Nerve;
        }
        if (std::holds_alternative<AttackerGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Attacker;
        }
        if (std::holds_alternative<InjectorGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Injector;
        }
        if (std::holds_alternative<MuscleGenomeDescription>(*cellFunction)) {
            return Enums::CellFunction_Muscle;
        }
        if (std::holds_alternative<PlaceHolderGenomeDescription1>(*cellFunction)) {
            return Enums::CellFunction_Placeholder1;
        }
        if (std::holds_alternative<PlaceHolderGenomeDescription2>(*cellFunction)) {
            return Enums::CellFunction_Placeholder2;
        }
        return Enums::CellFunction_None;
    }
    template <typename CellFunctionDesc>
    CellGenomeDescription& setCellFunction(CellFunctionDesc const& value)
    {
        cellFunction = value;
        return *this;
    }
};

using GenomeDescription = std::vector<CellGenomeDescription>;
