#pragma once

#include "Colors.h"

template <typename T>
struct ZoneValue
{
    T value;
    bool enabled = false;

    bool operator==(ZoneValue<T> const&) const = default;
};

template <typename T>
struct PinnableZoneValue
{
    T value;
    bool enabled = false;
    bool pinned = false;

    bool operator==(PinnableZoneValue<T> const&) const = default;
};

template <typename T>
struct BaseParameter
{
    T value;

    bool operator==(BaseParameter<T> const&) const = default;
};

template <typename T>
struct EnableableBaseParameter
{};

template <typename T>
struct ZoneParameter
{};


template <typename T>
struct BaseZoneParameter
{
    T baseValue;
    ZoneValue<T> zoneValues[MAX_ZONES];

    bool operator==(BaseZoneParameter<T> const&) const = default;
};

template <typename T>
struct PinnableBaseZoneParameter
{
    bool baseValuePinned = false;
    PinnableZoneValue<T> zoneValues[MAX_ZONES];

    bool operator==(PinnableBaseZoneParameter<T> const&) const = default;
};

enum class LocationType
{
    Base,
    Zone,
    Source
};

struct FloatColorRGB
{
    float r = 0;
    float g = 0;
    float b = 0;

    bool operator==(FloatColorRGB const&) const = default;
};

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
