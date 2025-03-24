#pragma once

using Char64 = char[64];

using CellColoring = int;
enum CellColoring_
{
    CellColoring_None,
    CellColoring_CellColor,
    CellColoring_MutationId,
    CellColoring_MutationId_EveryCellType,
    CellColoring_LivingState,
    CellColoring_GenomeSize,
    CellColoring_SpecificCellType,
    CellColoring_EveryCellType
};

using CellDeathConsquences = int;
enum CellDeathConsquences_
{
    CellDeathConsquences_None,
    CellDeathConsquences_CreatureDies,
    CellDeathConsquences_DetachedPartsDie
};

using MotionType = int;
enum MotionType_
{
    MotionType_Fluid,
    MotionType_Collision
};
