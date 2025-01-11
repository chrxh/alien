#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/CellTypeConstants.h"


using namespace std::string_literals;

namespace Const
{
    std::vector const CellTypeStrings = {
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

    std::map<CellType, std::string> const CellTypeToStringMap = {
        {CellType_Constructor, "Constructor"},
        {CellType_Attacker, "Attacker"},
        {CellType_Injector, "Injector"},
        {CellType_Muscle, "Muscle"},
        {CellType_Oscillator, "Oscillator"},
        {CellType_Neuron, "Neuron"},
        {CellType_Sensor, "Sensor"},
        {CellType_Transmitter, "Transmitter"},
        {CellType_Defender, "Defender"},
        {CellType_Reconnector, "Reconnector"},
        {CellType_Detonator, "Detonator"},
        {CellType_None, "None"},
    };
}
