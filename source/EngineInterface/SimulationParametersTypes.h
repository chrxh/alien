#pragma once

using Char64 = char[64];

using CellColoring = int;
enum CellColoring_
{
    CellColoring_None,
    CellColoring_CellColor,
    CellColoring_MutationId,
    CellColoring_MutationId_AllCellFunctions,
    CellColoring_LivingState,
    CellColoring_GenomeSize,
    CellColoring_CellFunction,
    CellColoring_AllCellFunctions
};

using CellDeathConsquences = int;
enum CellDeathConsquences_
{
    CellDeathConsquences_None,
    CellDeathConsquences_CreatureDies,
    CellDeathConsquences_DetachedPartsDie
};
