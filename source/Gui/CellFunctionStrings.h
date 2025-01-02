#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/CellFunctionConstants.h"


using namespace std::string_literals;

namespace Const
{
    std::vector const CellFunctionStrings = {
        "Neuron"s,
        "Transmitter"s,
        "Constructor"s,
        "Sensor"s,
        "Oscillator"s,
        "Attacker"s,
        "Injector"s,
        "Muscle"s,
        "Defender"s,
        "Reconnector"s,
        "Detonator"s,
        "None"s};

    std::map<CellFunction, std::string> const CellFunctionToStringMap = {
        {CellFunction_Constructor, "Constructor"},
        {CellFunction_Attacker, "Attacker"},
        {CellFunction_Injector, "Injector"},
        {CellFunction_Muscle, "Muscle"},
        {CellFunction_Oscillator, "Oscillator"},
        {CellFunction_Neuron, "Neuron"},
        {CellFunction_Sensor, "Sensor"},
        {CellFunction_Transmitter, "Transmitter"},
        {CellFunction_Defender, "Defender"},
        {CellFunction_Reconnector, "Reconnector"},
        {CellFunction_Detonator, "Detonator"},
        {CellFunction_None, "None"},
    };
}
