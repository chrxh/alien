#pragma once

#include "EngineConstants.h"

namespace Const
{
    auto constexpr GenomeHeaderSize = 10;
    auto constexpr GenomeHeaderShapePos = 0;
    auto constexpr GenomeHeaderNumBranchesPos = 1;
    auto constexpr GenomeHeaderSeparationPos = 2;
    auto constexpr GenomeHeaderAlignmentPos = 3;
    auto constexpr GenomeHeaderStiffnessPos = 4;
    auto constexpr GenomeHeaderConstructionDistancePos = 5;
    auto constexpr GenomeHeaderNumRepetitionsPos = 6;
    auto constexpr GenomeHeaderConcatenationAngle1Pos = 7;
    auto constexpr GenomeHeaderConcatenationAngle2Pos = 8;
    auto constexpr GenomeHeaderFrontAnglePos = 9;

    auto constexpr CellAnglePos = 1;
    auto constexpr CellRequiredConnectionsPos = 3;
    auto constexpr CellExecutionNumberPos = 4;
    auto constexpr CellColorPos = 5;
    auto constexpr CellInputExecutionNumberPos = 6;
    auto constexpr CellOutputBlockedPos = 7;

    auto constexpr ConstructorConstructionAngle1Pos = 3;
    auto constexpr ConstructorConstructionAngle2Pos = 4;

    auto constexpr NeuronWeightBytes = MAX_CHANNELS * MAX_CHANNELS;
    auto constexpr NeuronWeightAndBiasBytes = NeuronWeightBytes + MAX_CHANNELS;
    auto constexpr NeuronBytes = NeuronWeightAndBiasBytes + MAX_CHANNELS;
    auto constexpr CellBasicBytesWithoutNeurons = 8;
    auto constexpr CellBasicBytes = CellBasicBytesWithoutNeurons + NeuronBytes;
    auto constexpr BaseBytes = 0;
    auto constexpr TransmitterBytes = 1;
    auto constexpr ConstructorFixedBytes = 5;
    auto constexpr SensorBytes = 6;
    auto constexpr OscillatorBytes = 2;
    auto constexpr AttackerBytes = 1;
    auto constexpr InjectorFixedBytes = 1;
    auto constexpr MuscleBytes = 4;
    auto constexpr DefenderBytes = 1; 
    auto constexpr ReconnectorBytes = 2;
    auto constexpr DetonatorBytes = 2;
}
