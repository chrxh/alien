#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/Enums.h"


using namespace std::string_literals;

namespace Const
{
    auto const CellFunctionStrings = std::vector{
        "Neuron"s,
        "Transmitter"s,
        "Constructor"s,
        "Sensor"s,
        "Nerve"s,
        "Attacker"s,
        "Injector"s,
        "Muscle"s,
        "Placeholder1"s,
        "Placeholder2"s,
        "None"s};

    std::map<Enums::CellFunction, std::string> const CellFunctionToStringMap = {
        {Enums::CellFunction_Constructor, "Constructor"},
        {Enums::CellFunction_Attacker, "Attacker"},
        {Enums::CellFunction_Injector, "Injector"},
        {Enums::CellFunction_Muscle, "Muscle"},
        {Enums::CellFunction_Nerve, "Nerve"},
        {Enums::CellFunction_Neuron, "Neuron"},
        {Enums::CellFunction_Sensor, "Sensor"},
        {Enums::CellFunction_Transmitter, "Transmitter"},
        {Enums::CellFunction_Placeholder1, "Placeholder1"},
        {Enums::CellFunction_Placeholder2, "Placeholder2"},
        {Enums::CellFunction_None, "None"},
    };
}
