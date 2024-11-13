#pragma once

#include <cstdint>

using CellFunction = int;
enum CellFunction_
{
    CellFunction_Neuron,
    CellFunction_Transmitter,
    CellFunction_Constructor,
    CellFunction_Sensor,
    CellFunction_Nerve,
    CellFunction_Attacker,
    CellFunction_Injector,
    CellFunction_Muscle,
    CellFunction_Defender,
    CellFunction_Reconnector,
    CellFunction_Detonator,
    CellFunction_WithoutNone_Count,

    CellFunction_None = CellFunction_WithoutNone_Count,
    CellFunction_Count,
};

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

using NeuronActivationFunction = int;
enum NeuronActivationFunction_
{
    NeuronActivationFunction_Sigmoid,
    NeuronActivationFunction_BinaryStep,
    NeuronActivationFunction_Identity,
    NeuronActivationFunction_Abs,
    NeuronActivationFunction_Gaussian,
    NeuronActivationFunction_Count
};

using SensorMode = int;
enum SensorMode_
{
    SensorMode_Neighborhood,
    SensorMode_FixedAngle,
    SensorMode_Count
};

using SensorRestrictToMutants = int;
enum SensorRestrictToMutants_
{
    SensorRestrictToMutants_NoRestriction,
    SensorRestrictToMutants_RestrictToSameMutants,
    SensorRestrictToMutants_RestrictToOtherMutants,
    SensorRestrictToMutants_RestrictToFreeCells,
    SensorRestrictToMutants_RestrictToHandcraftedCells,
    SensorRestrictToMutants_RestrictToLessComplexMutants,
    SensorRestrictToMutants_RestrictToMoreComplexMutants,
    SensorRestrictToMutants_Count
};

using EnergyDistributionMode = int;
enum EnergyDistributionMode_
{
    EnergyDistributionMode_ConnectedCells,
    EnergyDistributionMode_TransmittersAndConstructors,
    EnergyDistributionMode_Count
};

using MuscleMode = int;
enum MuscleMode_
{
    MuscleMode_Movement,
    MuscleMode_ContractionExpansion,
    MuscleMode_Bending,
    MuscleMode_Count
};

using MuscleBendingDirection = int;
enum MuscleBendingDirection_
{
    MuscleBendingDirection_None,
    MuscleBendingDirection_Positive,
    MuscleBendingDirection_Negative
};

using DefenderMode = int;
enum DefenderMode_
{
    DefenderMode_DefendAgainstAttacker,
    DefenderMode_DefendAgainstInjector,
    DefenderMode_Count
};

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

using InjectorMode = int;
enum InjectorMode_
{
    InjectorMode_InjectOnlyEmptyCells,
    InjectorMode_InjectAll,
    InjectorMode_Count
};

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

using DetonatorState = int;
enum DetonatorState_
{
    DetonatorState_Ready,
    DetonatorState_Activated,
    DetonatorState_Exploded
};

using ReconnectorRestrictToMutants = int;
enum ReconnectorRestrictToMutants_
{
    ReconnectorRestrictToMutants_NoRestriction,
    ReconnectorRestrictToMutants_RestrictToSameMutants,
    ReconnectorRestrictToMutants_RestrictToOtherMutants,
    ReconnectorRestrictToMutants_RestrictToFreeCells,
    ReconnectorRestrictToMutants_RestrictToHandcraftedCells,
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

using CellFunctionUsed = uint8_t;
enum CellFunctionUsed_
{
    CellFunctionUsed_No,
    CellFunctionUsed_Yes,
};

using SignalOrigin = uint8_t;
enum SignalOrigin_
{
    SignalOrigin_Unknown,
    SignalOrigin_Sensor
};

auto constexpr MaxActivationTime = 256 * 4;