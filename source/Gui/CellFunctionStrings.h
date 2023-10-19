#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/CellFunctionConstants.h"


using namespace std::string_literals;

namespace Const
{
    std::vector const CellFunctionStrings =
        {"Neuron"s, "Transmitter"s, "Constructor"s, "Sensor"s, "Nerve"s, "Attacker"s, "Injector"s, "Muscle"s, "Defender"s, "Reconnector"s, "None"s};

    std::map<CellFunction, std::string> const CellFunctionToStringMap = {
        {CellFunction_Constructor, "Constructor"},
        {CellFunction_Attacker, "Attacker"},
        {CellFunction_Injector, "Injector"},
        {CellFunction_Muscle, "Muscle"},
        {CellFunction_Nerve, "Nerve"},
        {CellFunction_Neuron, "Neuron"},
        {CellFunction_Sensor, "Sensor"},
        {CellFunction_Transmitter, "Transmitter"},
        {CellFunction_Defender, "Defender"},
        {CellFunction_Reconnector, "Reconnector"},
        {CellFunction_None, "None"},
    };
}
