#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <map>

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
    CellType_Oscillator,
    CellType_Attacker,
    CellType_Injector,
    CellType_Muscle,
    CellType_Defender,
    CellType_Reconnector,
    CellType_Detonator,
    CellType_Count,
};

namespace Const
{
    std::vector<std::string> const CellTypeStrings =
        {"Structure", "Free", "Base", "Depot", "Constructor", "Sensor", "Oscillator", "Attacker", "Injector", "Muscle", "Defender", "Reconnector", "Detonator"};
}

using CellTypeGenome = int;
enum CellTypeGenome_
{
    CellTypeGenome_Base,
    CellTypeGenome_Depot,
    CellTypeGenome_Constructor,
    CellTypeGenome_Sensor,
    CellTypeGenome_Oscillator,
    CellTypeGenome_Attacker,
    CellTypeGenome_Injector,
    CellTypeGenome_Muscle,
    CellTypeGenome_Defender,
    CellTypeGenome_Reconnector,
    CellTypeGenome_Detonator,
    CellTypeGenome_Count,
};

namespace Const
{
    std::vector<std::string> const CellTypeGenomeStrings =
        {"Base", "Depot", "Constructor", "Sensor", "Oscillator", "Attacker", "Injector", "Muscle", "Defender", "Reconnector", "Detonator"};
}

using LivingState = int;
enum LivingState_
{
    LivingState_Ready,
    LivingState_UnderConstruction,
    LivingState_Activating,
    LivingState_Detaching,
    LivingState_Reviving,
    LivingState_Dying,
    LivingState_Count
};

using ActivationFunction = uint8_t;
enum ActivationFunction_
{
    ActivationFunction_Sigmoid,
    ActivationFunction_BinaryStep,
    ActivationFunction_Identity,
    ActivationFunction_Abs,
    ActivationFunction_Gaussian,
    ActivationFunction_Count
};

namespace Const
{
    std::vector<std::string> const ActivationFunctionStrings = {"Sigmoid", "Binary step", "Identity", "Absolute value", "Gaussian"};
}

//************************
//* Oscillator constants *
//************************
namespace Channels
{
    auto constexpr OscillatorPulse = 0;
}

//********************
//* Sensor constants *
//********************
namespace Channels
{
    auto constexpr SensorFoundResult = 0;
    auto constexpr SensorAngle = 1;
    auto constexpr SensorDensity = 2;
    auto constexpr SensorDistance = 3;
}

using SensorRestrictToMutants = int;
enum SensorRestrictToMutants_
{
    SensorRestrictToMutants_NoRestriction,
    SensorRestrictToMutants_RestrictToSameMutants,
    SensorRestrictToMutants_RestrictToOtherMutants,
    SensorRestrictToMutants_RestrictToFreeCells,
    SensorRestrictToMutants_RestrictToStructures,
    SensorRestrictToMutants_RestrictToLessComplexMutants,
    SensorRestrictToMutants_RestrictToMoreComplexMutants,
    SensorRestrictToMutants_Count
};

namespace Const
{
    std::vector<std::string> const SensorRestrictToMutantStrings =
        {"None", "Same mutants", "Other mutants", "Free cells", "Handcrafted cells", "Less complex mutants", "More complex mutants"};
}

//********************
//* Muscle constants *
//********************
namespace Channels
{
    auto constexpr AttackerNotify = 7;
    auto constexpr AttackerSuccess = 2;

    auto constexpr MuscleTrigger = 0;
    auto constexpr MuscleAngle = 1;
}

using MuscleMode = uint8_t;
enum MuscleMode_
{
    MuscleMode_AutoBending,
    MuscleMode_ManualBending,
    MuscleMode_AngleBending,
    MuscleMode_AutoCrawling,
    MuscleMode_ManualCrawling,
    MuscleMode_DirectMovement,
    MuscleMode_Count
};

//**********************
//* Defender constants *
//**********************
using DefenderMode = int;
enum DefenderMode_
{
    DefenderMode_DefendAgainstAttacker,
    DefenderMode_DefendAgainstInjector,
    DefenderMode_Count
};

//*************************
//* Constructor constants *
//*************************
using ConstructorAngleAlignment = int;
enum ConstructorAlignment_
{
    ConstructorAngleAlignment_None = 0,
    ConstructorAngleAlignment_180 = 1,
    ConstructorAngleAlignment_120 = 2,
    ConstructorAngleAlignment_90 = 3,
    ConstructorAngleAlignment_72 = 4,
    ConstructorAngleAlignment_60 = 5,
    ConstructorAngleAlignment_Count = 6
};

namespace Const
{
    std::vector<std::string> const ConstructorAlignmentStrings = {"None"s, "180 deg"s, "120 deg"s, "90 deg"s, "72 deg"s, "60 deg"s};
}

using ConstructionShape = int;
enum ConstructionShape_
{
    ConstructionShape_Custom,
    ConstructionShape_Segment,
    ConstructionShape_Triangle,
    ConstructionShape_Rectangle,
    ConstructionShape_Hexagon,
    ConstructionShape_Loop,
    ConstructionShape_Tube,
    ConstructionShape_Lolli,
    ConstructionShape_SmallLolli,
    ConstructionShape_Zigzag,
    ConstructionShape_Count
};

namespace Const
{
    std::vector<std::string> const ConstructionShapeStrings =
        {"Custom", "Segment", "Triangle", "Rectangle", "Hexagon", "Loop", "Tube", "Lolli", "Small Lolli", "Zigzag"};
}

//**********************
//* Injector constants *
//**********************
using InjectorMode = int;
enum InjectorMode_
{
    InjectorMode_InjectOnlyEmptyCells,
    InjectorMode_InjectAll,
    InjectorMode_Count
};

//***********************
//* Detonator constants *
//***********************
using DetonatorState = int;
enum DetonatorState_
{
    DetonatorState_Ready,
    DetonatorState_Activated,
    DetonatorState_Exploded
};

//*************************
//* Reconnector constants *
//*************************
using ReconnectorRestrictToMutants = int;
enum ReconnectorRestrictToMutants_
{
    ReconnectorRestrictToMutants_NoRestriction,
    ReconnectorRestrictToMutants_RestrictToSameMutants,
    ReconnectorRestrictToMutants_RestrictToOtherMutants,
    ReconnectorRestrictToMutants_RestrictToFreeCells,
    ReconnectorRestrictToMutants_RestrictToStructures,
    ReconnectorRestrictToMutants_RestrictToLessComplexMutants,
    ReconnectorRestrictToMutants_RestrictToMoreComplexMutants,
    ReconnectorRestrictToMutants_Count
};

using CellEvent = uint8_t;
enum CellEvent_
{
    CellEvent_No,
    CellEvent_Attacking,
    CellEvent_Attacked
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
    SignalOrigin_Sensor
};

using EnergyDistributionMode = int;
enum EnergyDistributionMode_
{
    EnergyDistributionMode_ConnectedCells,
    EnergyDistributionMode_TransmittersAndConstructors,
    EnergyDistributionMode_Count
};
