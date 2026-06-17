#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "EngineConstants.h"

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
    ActivationFunction_Mod,
    ActivationFunction_Count,
};

namespace Const
{
    std::vector<std::string> const ActivationFunctionStrings = {"Tanh", "Binary step", "Identity", "Absolute value", "Gaussian", "Mod"};
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
    auto constexpr ConstructorSuccess = 4;
}
namespace Const
{
    auto constexpr ConstructorAutoTriggerInterval_Min = 1;
    auto constexpr ConstructorAutoTriggerInterval_Default = 100;
    auto constexpr ConstructorConstructionActivationTime_Min = 0;
    auto constexpr ConstructorConstructionActivationTime_Max = 256 * 4;
    auto constexpr ConstructorConstructionActivationTime_Default = 100;
    auto constexpr ConstructorConstructionAngle_Min = -180.0f;
    auto constexpr ConstructorConstructionAngle_Max = 180.0f;
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
    ProvideEnergy_ReduceCellEnergy = 0,
    ProvideEnergy_Free = 1,
    ProvideEnergy_Count = 2,
};

namespace Const
{
    std::vector<std::string> const ConstructorShapeStrings = {"Segment", "Triangle", "Rectangle", "Hexagon", "Tube", "Large Lolli", "Small Lolli", "Zigzag"};
    std::vector<std::string> const ProvideEnergyStrings = {"Reduce cell energy", "Free"};
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
namespace Const
{
    auto constexpr GeneratorValue_Min = -2.0f;
    auto constexpr GeneratorValue_Max = 2.0f;
    auto constexpr GeneratorMinValue_Default = -1.0f;
    auto constexpr GeneratorMaxValue_Default = 1.0f;
    auto constexpr GeneratorAdditive_Default = false;
    auto constexpr GeneratorTimeOffset_Min = 0;
    auto constexpr GeneratorTimeOffset_Default = 0;
    auto constexpr GeneratorPeriod_Min = 1;
    auto constexpr GeneratorPeriod_Default = 100;
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
namespace Const
{
    auto constexpr DepotStorageLimit_Min = 0.0f;
    auto constexpr DepotStorageLimit_Max = 1000.0f;
    auto constexpr DepotStorageLimit_Default = 200.0f;
    auto constexpr DepotInitialStoredUsableEnergy_Min = 0.0f;
    auto constexpr DepotInitialStoredUsableEnergy_Default = 0.0f;
}

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
namespace Const
{
    auto constexpr SensorRange_Min = 0;
    auto constexpr SensorRange_Max = 512;
    auto constexpr SensorMinRange_Default = 0;
    auto constexpr SensorMaxRange_Default = 255;
    auto constexpr SensorAutoTrigger_Default = true;
    auto constexpr SensorTagForAttackers_Default = true;
    auto constexpr DetectEnergyMinDensity_Min = 0.0f;
    auto constexpr DetectEnergyMinDensity_Max = 1.0f;
    auto constexpr DetectEnergyMinDensity_Default = 0.05f;
    auto constexpr DetectFreeCellMinDensity_Min = 0.0f;
    auto constexpr DetectFreeCellMinDensity_Max = 1.0f;
    auto constexpr DetectFreeCellMinDensity_Default = 0.05f;
    auto constexpr RestrictToColors_Min = 0;
    auto constexpr RestrictToColors_Max = (1 << MAX_COLORS) - 1;
    auto constexpr RestrictToColors_Default = (1 << MAX_COLORS) - 1;
    auto constexpr CreatureNumCells_Min = 0;
}

using LineageRestriction = int;
enum LineageRestriction_
{
    LineageRestriction_No,
    LineageRestriction_RelatedLineage,
    LineageRestriction_UnrelatedLineage,
    LineageRestriction_Count,
};

using MutationType = int;
enum MutationType_
{
    MutationType_NoMutations,     // No mutations are applied to the genome
    MutationType_FixedMutations,  // Mutations are applied, but no meta-mutations (mutation rates stay fixed)
    MutationType_MetaMutations,   // Both mutations and meta-mutations are applied
    MutationType_Count,
};

namespace Const
{
    std::vector<std::string> const MutationTypeStrings = {"No mutations", "Fixed mutations", "Meta mutations"};
}

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
namespace Const
{
    auto constexpr MuscleModeRatio_Min = 0.0f;
    auto constexpr MuscleModeRatio_Max = 1.0f;
    auto constexpr MuscleMaxAngleDeviation_Default = 0.2f;
    auto constexpr MuscleForwardBackwardRatio_Default = 0.8f;
    auto constexpr MuscleAttractionRepulsionRatio_Default = 0.8f;
    auto constexpr MuscleMaxDistanceDeviation_Default = 0.8f;
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
namespace Const
{
    auto constexpr InjectorGeneIndex_Default = 0;
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
namespace Const
{
    auto constexpr DetonatorCountdown_Min = 1;
    auto constexpr DetonatorCountdown_Default = 10;
}

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
namespace Channels
{
    auto constexpr ReconnectorSuccess = 2;
}

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

//********************
//* Memory constants *
//********************
namespace Channels
{
    auto constexpr MemoryReadWriteAction = 0;
}
namespace Const
{
    auto constexpr DigestorRawEnergyConductivity_Min = 0.0f;
    auto constexpr DigestorRawEnergyConductivity_Max = 1.0f;
    auto constexpr DigestorRawEnergyConductivity_Default = 0.5f;
    auto constexpr MemoryChannelBitMask_Min = uint16_t{0};
    auto constexpr MemoryChannelBitMask_Max = uint16_t{0xffff};
    auto constexpr MemoryNumSignalEntries_Min = 0;
    auto constexpr MemoryNumSignalEntries_Max = MAX_CELL_MEMORY_ENTRIES;
    auto constexpr SignalDelay_Min = 0;
    auto constexpr SignalDelay_Max = MAX_CELL_MEMORY_ENTRIES;
    auto constexpr SignalDelay_Default = 10;
    auto constexpr SignalIntegratorNewSignalWeight_Min = 0.0f;
    auto constexpr SignalIntegratorNewSignalWeight_Max = 1.0f;
    auto constexpr SignalIntegratorNewSignalWeight_Default = 0.5f;
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
namespace Const
{
    auto constexpr CommunicatorRange_Min = 0;
    auto constexpr CommunicatorRange_Max = 20;
    auto constexpr CommunicatorRange_Default = 15;
    auto constexpr CommunicatorMaxTimesSent_Min = 0;
    auto constexpr CommunicatorMaxTimesSent_Default = 4;
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
