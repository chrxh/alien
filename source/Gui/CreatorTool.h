#pragma once

#include <Base/Singleton.h>

#include <Data/CreatorService.h>
#include <Data/DescEditService.h>
#include <Data/Descs.h>

#include "Definitions.h"
#include "EditorModel.h"
#include "MainLoopEntity.h"

class CreatorTool : public MainLoopEntity
{
    MAKE_SINGLETON(CreatorTool);

public:
    void processOptions();

    void onPlace(RealVector2D const& worldPos);

    void onAddPoint(RealVector2D const& worldPos);
    void onRemoveLastPoint();
    bool isFinishingPointsPossible() const;
    void onFinishPoints();
    bool hasPoints() const;
    void onAbortPoints();

    void onDrawing();
    void finishDrawing();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void processShapeWidgets();
    void processColorWidget();
    void processMaterialWidgets();
    void processObjectDistanceWidget();
    void processStickyWidget();
    void processStaticWidget();
    void processPointButtons();

    void processPlacementPreview();
    void processPointPreview(std::vector<RealVector2D> const& path, bool closed) const;
    void processControlPolygonPreview() const;

    ContentDesc createShape(RealVector2D const& center) const;
    void addToSimulation(ContentDesc&& content) const;

    void validateAndCorrect();
    bool isEnergyMaterial() const;
    int getMinNumPoints() const;

    CreatorService::ObjectProperties getObjectProperties() const;

    float _energy = 100.0f;
    float _stiffness = 1.0f;
    bool _static = false;
    float _objectDistance = 1.0f;
    float _glow = 0.0f;
    bool _makeSticky = false;

    int _rectHorizontalObjects = 10;
    int _rectVerticalObjects = 10;

    int _layers = 10;

    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    CreationMaterial _material = CreationMaterial_Solid;
    ContentDesc _drawingDescription;
    DescEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    std::vector<RealVector2D> _points;
    EditTool _lastTool = EditTool_Select;

    struct PreviewKey
    {
        EditTool tool = EditTool_Select;
        CreationMaterial material = CreationMaterial_Solid;
        int rectHorizontalObjects = 0;
        int rectVerticalObjects = 0;
        int layers = 0;
        float outerRadius = 0;
        float innerRadius = 0;
        float objectDistance = 0;

        bool operator==(PreviewKey const& other) const = default;
    };
    std::optional<PreviewKey> _previewKey;
    std::vector<RealVector2D> _previewPositions;
};
