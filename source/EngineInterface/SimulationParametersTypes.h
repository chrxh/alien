#pragma once

#include "Colors.h"

struct ExpertToggle
{
    bool value = false;

    bool operator==(ExpertToggle const&) const = default;
};

/**
 * Base parameters
 */

template <typename T>
struct BaseParameter
{
    T value;

    bool operator==(BaseParameter<T> const&) const = default;
};

template <typename T>
struct EnableableBaseParameter
{
    T value;
    bool enabled = false;

    bool operator==(EnableableBaseParameter<T> const&) const = default;
};

struct PinBaseParameter
{
    bool pinned = false;

    bool operator==(PinBaseParameter const&) const = default;
};

/**
 * Zone parameters
 */

template <typename T>
struct ZoneParameter
{
    T zoneValues[MAX_ZONES];

    bool operator==(ZoneParameter<T> const&) const = default;
};

template <typename T>
struct EnableableValue
{
    T value;
    bool enabled = false;

    bool operator==(EnableableValue<T> const&) const = default;
};

/**
 * Base and zone parameters
 */

template <typename T>
struct BaseZoneParameter
{
    T baseValue;
    EnableableValue<T> zoneValues[MAX_ZONES];

    bool operator==(BaseZoneParameter<T> const&) const = default;
};

/**
 * Source parameters
 */

template <typename T>
struct SourceParameter
{
    T sourceValues[MAX_ZONES];

    bool operator==(SourceParameter<T> const&) const = default;
};

template <typename T>
struct PinnableSourceValue
{
    T value;
    bool pinned = false;

    bool operator==(PinnableSourceValue<T> const&) const = default;
};

template <typename T>
struct PinnableSourceParameter
{
    PinnableSourceValue<T> value[MAX_RADIATION_SOURCES];

    bool operator==(PinnableSourceParameter<T> const&) const = default;
};

/**
 * Misc
 */

struct ColorTransitionRules
{
    ColorVector<int> cellColorTransitionDuration = {
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value,
        Infinity<int>::value};
    ColorVector<int> cellColorTransitionTargetColor = {0, 1, 2, 3, 4, 5, 6};

    bool operator==(ColorTransitionRules const&) const = default;
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

using Orientation = int;
enum Orientation_
{
    Orientation_Clockwise,
    Orientation_CounterClockwise
};

using ForceField = int;
enum ForceField_
{
    ForceField_None,
    ForceField_Radial,
    ForceField_Central,
    ForceField_Linear
};

using ZoneShapeType = int;
enum ZoneShapeType_
{
    ZoneShapeType_Circular,
    ZoneShapeType_Rectangular
};
