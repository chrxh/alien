#pragma once

namespace Const
{
    auto constexpr CellFunctionMutationMaxGenomeSize = 200;

    auto constexpr GenomeHeaderSize = 9;
    auto constexpr GenomeHeaderShapePos = 0;
    auto constexpr GenomeHeaderNumBranchesPos = 1;
    auto constexpr GenomeHeaderSeparationPos = 2;
    auto constexpr GenomeHeaderAlignmentPos = 3;
    auto constexpr GenomeHeaderStiffnessPos = 4;
    auto constexpr GenomeHeaderConstructionDistancePos = 5;
    auto constexpr GenomeHeaderNumRepetitionsPos = 6;
    auto constexpr GenomeHeaderConcatenationAngle1Pos = 7;
    auto constexpr GenomeHeaderConcatenationAngle2Pos = 8;

    auto constexpr CellAnglePos = 1;
    auto constexpr CellRequiredConnectionsPos = 3;
    auto constexpr CellColorPos = 5;

    auto constexpr ConstructorConstructionAngle1Pos = 3;
    auto constexpr ConstructorConstructionAngle2Pos = 4;

    auto constexpr CellBasicBytes = 8;
    auto constexpr NeuronBytes = 64 + 8 + 8;
    auto constexpr TransmitterBytes = 1;
    auto constexpr ConstructorFixedBytes = 5;
    auto constexpr SensorBytes = 5;
    auto constexpr NerveBytes = 2;
    auto constexpr AttackerBytes = 1;
    auto constexpr InjectorFixedBytes = 1;
    auto constexpr MuscleBytes = 1;
    auto constexpr DefenderBytes = 1;
    auto constexpr ReconnectorBytes = 2;
    auto constexpr DetonatorBytes = 2;
}
