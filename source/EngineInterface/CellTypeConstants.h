#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace std::string_literals;

//***********
//* General *
//***********
namespace Channels
{
    auto constexpr CellTypeActivation = 0;
}

using ObjectType = int;
enum ObjectType_
{
    ObjectType_Solid,
    ObjectType_Fluid,
    ObjectType_FreeCell,
    ObjectType_Cell,
    ObjectType_Count,
};

using CellType = int;
enum CellType_
{
    CellType_Base,
    CellType_Depot,
    CellType_Sensor,
    CellType_Generator,
    CellType_Attacker,
    CellType_Injector,
    CellType_Muscle,
    CellType_Defender,
    CellType_Reconnector,
    CellType_Detonator,
    CellType_Digestor,
    CellType_Memory,
    CellType_Communicator,
    CellType_Void,
    CellType_Count,
};

namespace Const
{
    std::vector<std::string> const CellTypeStrings = {
        "Base",
        "Depot",
        "Sensor",
        "Generator",
        "Attacker",
        "Injector",
        "Muscle",
        "Defender",
        "Reconnector",
        "Detonator",
        "Digestor",
        "Memory",
        "Communicator",
        "Void"};

    std::vector<std::string> const ObjectTypeStrings = {"Solid", "Fluid", "Free Cell", "Cell"};
}


using CellState = int;
enum CellState_
{
    CellState_Ready,
    CellState_Constructing,
    CellState_Activating,
    CellState_Dying,
    CellState_InstantDying,
    CellState_Count,
};

using ActivationFunction = uint8_t;
enum ActivationFunction_
{
    ActivationFunction_Tanh,
    ActivationFunction_BinaryStep,
    ActivationFunction_Identity,
    ActivationFunction_Abs,
    ActivationFunction_Gaussian,
    ActivationFunction_Count,
};

namespace Const
{
    std::vector<std::string> const ActivationFunctionStrings = {"Tanh", "Binary step", "Identity", "Absolute value", "Gaussian"};
}

using CellEvent = uint8_t;
enum CellEvent_
{
    CellEvent_No,
    CellEvent_Attacking,
    CellEvent_Attacked,
    CellEvent_Detonation,
};

using SignalOrigin = uint8_t;
enum SignalOrigin_
{
    SignalOrigin_Unknown,
    SignalOrigin_Sensor,
};

//*************************
//* Constructor constants *
//*************************
namespace Channels
{
    auto constexpr ConstructorSuccess = 1;
}

using ConstructorShape = int;
enum ConstructorShape_
{
    ConstructorShape_Segment,
    ConstructorShape_Triangle,
    ConstructorShape_Rectangle,
    ConstructorShape_Hexagon,
    ConstructorShape_Tube,
    ConstructorShape_LargeLolli,
    ConstructorShape_SmallLolli,
    ConstructorShape_Zigzag,
    ConstructorShape_Count,
};

using ProvideEnergy = uint8_t;
enum ProvideEnergy_
{
    ProvideEnergy_CellOnly = 0,
    ProvideEnergy_FreeGeneration = 1,
    ProvideEnergy_Count = 2,
};

namespace Const
{
    std::vector<std::string> const ConstructorShapeStrings =
        {"Segment", "Triangle", "Rectangle", "Hexagon", "Tube", "Large Lolli", "Small Lolli", "Zigzag"};
}

//**********************
//* Mutation constants *
//**********************
using MutationState = int;
enum MutationState_
{
    MutationState_NotMutated,
    MutationState_MutationInProgress,
    MutationState_Mutated,
};


//************************
//* Generator constants *
//************************
namespace Channels
{
    auto constexpr GeneratorOutput = 0;
}

using GeneratorMode = int;
enum GeneratorMode_
{
    GeneratorMode_SquareSignal,
    GeneratorMode_SawtoothSignal,
    GeneratorMode_Count,
};

namespace Const
{
    std::vector<std::string> const GeneratorModeStrings = {"Square signal", "Sawtooth signal"};
}

//*******************
//* Depot constants *
//*******************

//********************
//* Sensor constants *
//********************
namespace Channels
{
    auto constexpr SensorWithRelocationScan = 0;
    auto constexpr SensorFoundResult = 0;
    auto constexpr SensorAngle = 1;
    auto constexpr SensorMass = 2;  // numCells for SensorMode_DetectCreature and density for other modes except SensorMode_Telemetry
    auto constexpr SensorDistance = 3;
    auto constexpr SensorTelemetryCellEnergy = 1;
    auto constexpr SensorTelemetryCellVelAngle = 2;
    auto constexpr SensorTelemetryCellVelStrength = 3;
}

using LineageRestriction = int;
enum LineageRestriction_
{
    LineageRestriction_No,
    LineageRestriction_RelatedLineage,
    LineageRestriction_UnrelatedLineage,
    LineageRestriction_Count,
};

using SensorMode = int;
enum SensorMode_
{
    SensorMode_Telemetry,
    SensorMode_DetectEnergy,
    SensorMode_DetectSolid,
    SensorMode_DetectFreeCell,
    SensorMode_DetectCreature,
    SensorMode_Count,
};

namespace Const
{
    std::vector<std::string> const SensorModeStrings = {"Telemetry", "Detect energy", "Detect solid", "Detect free cell", "Detect creature"};
}

//********************
//* Muscle constants *
//********************
namespace Channels
{
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
    auto constexpr AttackerSuccess = 2;
    auto constexpr AttackerNotify = 7;
}

using AttackerMode = int;
enum AttackerMode_
{
    AttackerMode_FreeCell,
    AttackerMode_Creature,
    AttackerMode_Count,
};

namespace Const
{
    std::vector<std::string> const AttackerModeStrings = {"Free cell", "Creature"};
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
    ReconnectorMode_Solid,
    ReconnectorMode_FreeCell,
    ReconnectorMode_Creature,
    ReconnectorMode_Count,
};

namespace Const
{
    std::vector<std::string> const ReconnectorModeStrings = {"Solid", "Free cell", "Creature"};
}

namespace Channels
{
    auto constexpr ReconnectorSuccess = 2;
}

//********************
//* Memory constants *
//********************
namespace Channels
{
    auto constexpr MemoryReadWriteAction = 0;
}

using MemoryMode = int;
enum MemoryMode_
{
    MemoryMode_SignalDelay,
    MemoryMode_SignalRecorder,
    MemoryMode_SignalStorage,
    MemoryMode_SignalIntegrator,
    MemoryMode_Count,
};

namespace Const
{
    std::vector<std::string> const MemoryModeStrings = {"Signal delay", "Signal recorder", "Signal storage", "Signal integrator"};
}

using SignalRecorderState = uint8_t;
enum SignalRecorderState_
{
    SignalRecorderState_Idle,
    SignalRecorderState_Recording,
    SignalRecorderState_Reading,
};

//***************************
//* Communicator constants *
//***************************
namespace Channels
{
    auto constexpr CommunicatorAngle = 1;
}

using CommunicatorMode = int;
enum CommunicatorMode_
{
    CommunicatorMode_Sender,
    CommunicatorMode_Receiver,
    CommunicatorMode_Count,
};

namespace Const
{
    std::vector<std::string> const CommunicatorModeStrings = {"Send", "Receive"};
}
