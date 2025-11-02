#pragma once

#include <limits>

#include "Colors.h"

template <typename T>
struct Infinity
{
    static auto constexpr value = std::numeric_limits<T>::max();
};

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
 * Layer parameters
 */

template <typename T>
struct LayerParameter
{
    T layerValues[MAX_LAYERS];

    bool operator==(LayerParameter<T> const&) const = default;
};

template <typename T>
struct EnableableValue
{
    T value;
    bool enabled = false;

    bool operator==(EnableableValue<T> const&) const = default;
};

template <typename T>
struct EnableableLayerParameter
{
    EnableableValue<T> layerValues[MAX_LAYERS];

    bool operator==(EnableableLayerParameter<T> const&) const = default;
};

/**
 * Base and layer parameters
 */

template <typename T>
struct BaseLayerParameter
{
    T baseValue;
    EnableableValue<T> layerValues[MAX_LAYERS];

    bool operator==(BaseLayerParameter<T> const&) const = default;
};

/**
 * Source parameters
 */

template <typename T>
struct SourceParameter
{
    T sourceValues[MAX_SOURCES];

    bool operator==(SourceParameter<T> const&) const = default;
};

template <typename T>
struct EnableableSourceParameter
{
    EnableableValue<T> sourceValues[MAX_SOURCES];

    bool operator==(EnableableSourceParameter<T> const&) const = default;
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
    PinnableSourceValue<T> sourceValues[MAX_SOURCES];

    bool operator==(PinnableSourceParameter<T> const&) const = default;
};

/**
 * Misc
 */

struct ColorTransitionRule
{
    int duration = Infinity<int>::value;
    int targetColor = 0;

    bool operator==(ColorTransitionRule const&) const = default;
};

enum class LocationType
{
    Base,
    Layer,
    Source
};

using Char64 = char[64];

using CellColoring = int;
enum CellColoring_
{
    CellColoring_None,
    CellColoring_CellColor,
    CellColoring_LineageId,
    CellColoring_LineageId_EveryCellType,
    CellColoring_CellState,
    CellColoring_NumCells,
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

using LayerShapeType = int;
enum LayerShapeType_
{
    LayerShapeType_Circular,
    LayerShapeType_Rectangular
};

using SourceShapeType = int;
enum SourceShapeType_
{
    SourceShapeType_Circular,
    SourceShapeType_Rectangular
};
