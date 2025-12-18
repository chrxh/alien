#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace std::string_literals;

using CellType = int;
enum CellType_
{
    CellType_Structure,
    CellType_Free,
    CellType_Base,
    CellType_Depot,
    CellType_Constructor,
    CellType_Sensor,
    CellType_Generator,
    CellType_Attacker,
    CellType_Injector,
    CellType_Muscle,
    CellType_Defender,
    CellType_Reconnector,
    CellType_Detonator,
    CellType_Digestor,
    CellType_Count,
};

namespace Const
{
    std::vector<std::string> const CellTypeStrings =
        {"Structure", "Free", "Base", "Depot", "Constructor", "Sensor", "Generator", "Attacker", "Injector", "Muscle", "Defender", "Reconnector", "Detonator", "Digestor"};
}

using CellTypeGenome = int;
enum CellTypeGenome_
{
    CellTypeGenome_Base,
    CellTypeGenome_Depot,
    CellTypeGenome_Constructor,
    CellTypeGenome_Sensor,
    CellTypeGenome_Generator,
    CellTypeGenome_Attacker,
    CellTypeGenome_Injector,
    CellTypeGenome_Muscle,
    CellTypeGenome_Defender,
    CellTypeGenome_Reconnector,
    CellTypeGenome_Detonator,
    CellTypeGenome_Digestor,
    CellTypeGenome_Count,
};

namespace Const
{
    std::vector<std::string> const CellTypeGenomeStrings =
        {"Base", "Depot", "Constructor", "Sensor", "Generator", "Attacker", "Injector", "Muscle", "Defender", "Reconnector", "Detonator", "Digestor"};
}

using CellState = int;
enum CellState_
{
    CellState_Ready,
    CellState_Constructing,
    CellState_Activating,
    CellState_Detaching,
    CellState_Reviving,
    CellState_Dying,
    CellState_Count,
};

using ActivationFunction = uint8_t;
enum ActivationFunction_
{
    ActivationFunction_Sigmoid,
    ActivationFunction_BinaryStep,
    ActivationFunction_Identity,
    ActivationFunction_Abs,
    ActivationFunction_Gaussian,
    ActivationFunction_Count,
};

using SignalState = uint8_t;
enum SignalState_
{
    SignalState_Inactive,
    SignalState_Fading,
    SignalState_Active,
    SignalState_Count,
};

namespace Const
{
    std::vector<std::string> const ActivationFunctionStrings = {"Sigmoid", "Binary step", "Identity", "Absolute value", "Gaussian"};
}

//************************
//* Generator constants *
//************************
namespace Channels
{
    auto constexpr GeneratorPulse = 0;
}

//*******************
//* Depot constants *
//*******************
namespace Channels
{
    auto constexpr DepotActivation = 0;
}

//********************
//* Sensor constants *
//********************
namespace Channels
{
    auto constexpr SensorForceInitialScan = 0;
    auto constexpr SensorFoundResult = 0;
    auto constexpr SensorAngle = 1;
    auto constexpr SensorDensity = 2;
    auto constexpr SensorDistance = 3;
    auto constexpr SensorTelemetryCellEnergy = 1;
    auto constexpr SensorTelemetryCellVelAngle = 2;
    auto constexpr SensorTelemetryCellVelStrength = 3;
}

using DetectCreatureLineageRestriction = int;
enum DetectCreatureLineageRestriction_
{
    DetectCreatureLineageRestriction_No,
    DetectCreatureLineageRestriction_SameLineage,
    DetectCreatureLineageRestriction_OtherLineage,
    DetectCreatureLineageRestriction_Count,
};

using SensorMode = int;
enum SensorMode_
{
    SensorMode_Telemetry,
    SensorMode_DetectEnergy,
    SensorMode_DetectStructure,
    SensorMode_DetectFreeCell,
    SensorMode_DetectCreature,
    SensorMode_Count,
};

namespace Const
{
    std::vector<std::string> const SensorModeStrings =
        {"Telemetry", "Detect energy", "Detect structure", "Detect free cell", "Detect creature"};
}

//********************
//* Muscle constants *
//********************
namespace Channels
{
    auto constexpr MuscleTrigger = 0;
    auto constexpr MuscleAngle = 1;
}

using MuscleMode = int;
enum MuscleMode_
{
    MuscleMode_AutoBending,
    MuscleMode_ManualBending,
    MuscleMode_AngleBending,
    MuscleMode_AutoCrawling,
    MuscleMode_ManualCrawling,
    MuscleMode_DirectMovement,
    MuscleMode_Count,
};

namespace Const
{
    std::vector<std::string> const MuscleModeStrings =
        {"Auto bending", "Manual bending", "Angle bending", "Auto crawling", "Manual crawling", "Direct movement"};
}

//**********************
//* Attacker constants *
//**********************
namespace Channels
{
    auto constexpr AttackerSuccess = 1;
    auto constexpr AttackerNotify = 7;
}

//**********************
//* Defender constants *
//**********************
using DefenderMode = int;
enum DefenderMode_
{
    DefenderMode_DefendAgainstAttacker,
    DefenderMode_DefendAgainstInjector,
    DefenderMode_Count,
};

namespace Const
{
    std::vector<std::string> const DefenderModeStrings = {"Anti-attacker", "Anti-injector"};
}

//*************************
//* Constructor constants *
//*************************
namespace Channels
{
    auto constexpr ConstructorSuccess = 1;
}

using ConstructorAngleAlignment = int;
enum ConstructorAlignment_
{
    ConstructorAngleAlignment_None = 0,
    ConstructorAngleAlignment_180 = 1,
    ConstructorAngleAlignment_120 = 2,
    ConstructorAngleAlignment_90 = 3,
    ConstructorAngleAlignment_72 = 4,
    ConstructorAngleAlignment_60 = 5,
    ConstructorAngleAlignment_Count = 6,
};

namespace Const
{
    std::vector<std::string> const ConstructorAlignmentStrings = {"None"s, "180 deg"s, "120 deg"s, "90 deg"s, "72 deg"s, "60 deg"s};
}

using ConstructorShape = int;
enum ConstructorShape_
{
    ConstructorShape_Custom,
    ConstructorShape_Segment,
    ConstructorShape_Triangle,
    ConstructorShape_Rectangle,
    ConstructorShape_Hexagon,
    ConstructorShape_Loop,
    ConstructorShape_Tube,
    ConstructorShape_Lolli,
    ConstructorShape_SmallLolli,
    ConstructorShape_Zigzag,
    ConstructorShape_Count,
};

using ProvideEnergy = uint8_t;
enum ProvideEnergy_
{
    ProvideEnergy_CellOnly,
    ProvideEnergy_CellAndGene,
    ProvideEnergy_FreeGeneration,
};

namespace Const
{
    std::vector<std::string> const ConstructorShapeStrings =
        {"Custom", "Segment", "Triangle", "Rectangle", "Hexagon", "Loop", "Tube", "Lolli", "Small Lolli", "Zigzag"};
}

//************************
//* Generator constants *
//************************
using GeneratorPulseType = int;
enum GeneratorPulseType_
{
    GeneratorPulseType_Positive,
    GeneratorPulseType_Alternation,
    GeneratorPulseType_Count,
};

//**********************
//* Injector constants *
//**********************
namespace Channels
{
    auto constexpr InjectorSuccess = 1;
}

using InjectorMode = int;
enum InjectorMode_
{
    InjectorMode_InjectOnlyEmptyCells,
    InjectorMode_InjectAll,
    InjectorMode_Count,
};

namespace Const
{
    std::vector<std::string> const InjectorModeStrings = {"Only empty cells", "All cells"};
}

//***********************
//* Detonator constants *
//***********************
using DetonatorState = int;
enum DetonatorState_
{
    DetonatorState_Ready,
    DetonatorState_Activated,
    DetonatorState_Exploded,
};

//*************************
//* Reconnector constants *
//*************************
using ReconnectorMode = int;
enum ReconnectorMode_
{
    ReconnectorMode_Structure,
    ReconnectorMode_FreeCell,
    ReconnectorMode_Creature,
    ReconnectorMode_Count,
};

namespace Const
{
    std::vector<std::string> const ReconnectorModeStrings = {"Structure", "Free cell", "Creature"};
}

using CellEvent = uint8_t;
enum CellEvent_
{
    CellEvent_No,
    CellEvent_Attacking,
    CellEvent_Attacked,
    CellEvent_Detonation,
};

using CellTriggered = uint8_t;
enum CellTriggered_
{
    CellTriggered_No,
    CellTriggered_Yes,
};

using SignalOrigin = uint8_t;
enum SignalOrigin_
{
    SignalOrigin_Unknown,
    SignalOrigin_Sensor,
};
