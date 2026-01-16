#pragma once

#include <map>
#include <string>
#include <vector>

#include <EngineInterface/CellTypeConstants.h>


using namespace std::string_literals;

namespace Const
{
    std::vector const CellTypeStrings = {
        "Base"s,
        "Depot"s,
        "Constructor"s,
        "Sensor"s,
        "Generator"s,
        "Attacker"s,
        "Injector"s,
        "Muscle"s,
        "Defender"s,
        "Reconnector"s,
        "Detonator"s,
        "Digestor"s,
        "Memory"s,
        "Communicator"s};

    std::map<CellType, std::string> const CellTypeToStringMap = {
        {CellType_Base, "Base"},
        {CellType_Depot, "Depot"},
        {CellType_Constructor, "Constructor"},
        {CellType_Sensor, "Sensor"},
        {CellType_Generator, "Generator"},
        {CellType_Attacker, "Attacker"},
        {CellType_Injector, "Injector"},
        {CellType_Muscle, "Muscle"},
        {CellType_Defender, "Defender"},
        {CellType_Reconnector, "Reconnector"},
        {CellType_Detonator, "Detonator"},
        {CellType_Digestor, "Digestor"},
        {CellType_Memory, "Memory"},
        {CellType_Communicator, "Communicator"},
    };

    std::map<ObjectType, std::string> const ObjectTypeToStringMap = {
        {ObjectType_Structure, "Structure"},
        {ObjectType_FreeCell, "Free Cell"},
        {ObjectType_Cell, "Cell"},
    };
}
