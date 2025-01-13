#pragma once

#include <vector>
#include <string>
#include <map>

#include "EngineInterface/CellTypeConstants.h"


using namespace std::string_literals;

namespace Const
{
    std::vector const CellTypeStrings = {
        "Structure"s,
        "Free"s,
        "Base"s,
        "Depot"s,
        "Constructor"s,
        "Sensor"s,
        "Oscillator"s,
        "Attacker"s,
        "Injector"s,
        "Muscle"s,
        "Defender"s,
        "Reconnector"s,
        "Detonator"s};

    std::map<CellType, std::string> const CellTypeToStringMap = {
        {CellType_Structure, "Structure"},
        {CellType_Free, "Free"},
        {CellType_Base, "Base"},
        {CellType_Depot, "Depot"},
        {CellType_Constructor, "Constructor"},
        {CellType_Sensor, "Sensor"},
        {CellType_Oscillator, "Oscillator"},
        {CellType_Attacker, "Attacker"},
        {CellType_Injector, "Injector"},
        {CellType_Muscle, "Muscle"},
        {CellType_Defender, "Defender"},
        {CellType_Reconnector, "Reconnector"},
        {CellType_Detonator, "Detonator"},
    };
}
