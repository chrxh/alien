#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/CellFunctionEnums.h"


using namespace std::string_literals;

namespace Const
{
    std::vector const CellFunctionStrings =
        {"Neuron"s, "Transmitter"s, "Constructor"s, "Sensor"s, "Nerve"s, "Attacker"s, "Injector"s, "Muscle"s, "Placeholder1"s, "Placeholder2"s, "None"s};

    std::map<CellFunction, std::string> const CellFunctionToStringMap = {
        {CellFunction_Constructor, "Constructor"},
        {CellFunction_Attacker, "Attacker"},
        {CellFunction_Injector, "Injector"},
        {CellFunction_Muscle, "Muscle"},
        {CellFunction_Nerve, "Nerve"},
        {CellFunction_Neuron, "Neuron"},
        {CellFunction_Sensor, "Sensor"},
        {CellFunction_Transmitter, "Transmitter"},
        {CellFunction_Placeholder1, "Placeholder1"},
        {CellFunction_Placeholder2, "Placeholder2"},
        {CellFunction_None, "None"},
    };
}
