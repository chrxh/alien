#pragma once

namespace Enums
{
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
        CellFunction_Placeholder1,
        CellFunction_Placeholder2,
        CellFunction_WithoutNoneCount,

        CellFunction_None = CellFunction_WithoutNoneCount,
        CellFunction_Count,
    };

    using LivingState = int;
    enum LivingState_
    {
        LivingState_Ready,
        LivingState_UnderConstruction,
        LivingState_JustReady,
        LivingState_Dying,
        LivingState_Count
    };

    using SensorMode = int;
    enum SensorMode_
    {
        SensorMode_Neighborhood,
        SensorMode_FixedAngle,
        SensorMode_Count
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
        MuscleMode_Movement = 0,
        MuscleMode_ContractionExpansion = 1,
        MuscleMode_Bending = 2,
        MuscleMode_Count = 3
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

    /*
    using Sensor = int;
    enum Sensor_
    {
        Sensor_Output = 5,
        Sensor_Input = 20,
        Sensor_InOutAngle = 21,
        Sensor_InMinDensity = 22,
        Sensor_InMaxDensity = 23,
        Sensor_InColor = 8,
        Sensor_OutDensity = 24,
        Sensor_OutDistance = 25
    };
    using SensorIn = int;
    enum SensorIn_
    {
        SensorIn_DoNothing,
        SensorIn_SearchVicinity,
        SensorIn_SearchByAngle,
        SensorIn_Count
    };
    using SensorOut = int;
    enum SensorOut_
    {
        SensorOut_NothingFound,
        SensorOut_ClusterFound
    };
*/
}
