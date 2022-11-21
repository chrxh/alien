#pragma once

#include <vector>
#include <optional>
#include <variant>

#include "Constants.h"
#include "Enums.h"

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
    auto operator<=>(TransmitterGenomeDescription const&) const = default;
};

struct ConstructorGenomeDescription
{
    int mode = 1;   //0 = manual, 1 = every cycle, 2 = every second cycle, 3 = every third cycle, etc.
    bool singleConstruction = false;
    bool separateConstruction = true;
    bool adaptMaxConnections = true;
    Enums::ConstructorAngleAlignment angleAlignment = Enums::ConstructorAngleAlignment_60;

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
    ConstructorGenomeDescription& setAngleAlignment(Enums::ConstructorAngleAlignment value)
    {
        angleAlignment = value;
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
    std::optional<float> fixedAngle;   //nullotr = entire neighborhood
    float minDensity = 0.3f;
    int color = 0;

    auto operator<=>(SensorGenomeDescription const&) const = default;

    Enums::SensorMode getSensorMode() const { return fixedAngle.has_value() ? Enums::SensorMode_FixedAngle : Enums::SensorMode_Neighborhood; }
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
    float referenceDistance = 1.0f;
    float referenceAngle = 0;
    int color = 0;
    int maxConnections = 0;
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
