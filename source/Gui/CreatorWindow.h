#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/Descs.h>

#include "AlienWindow.h"
#include "Definitions.h"
#include "SimulationInteractionController.h"

using CreationMode = int;
enum CreationMode_
{
    CreationMode_CreateObject,
    CreationMode_CreateRectangle,
    CreationMode_CreateHexagon,
    CreationMode_CreateDisc,
    CreationMode_Drawing,
    CreationMode_CreateLine,
    CreationMode_CreateCurve,
    CreationMode_CreatePolygon
};

using CreationMaterial = int;
enum CreationMaterial_
{
    CreationMaterial_Solid,
    CreationMaterial_Fluid,
    CreationMaterial_FreeCell,
    CreationMaterial_EnergyParticle
};

class CreatorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatorWindow);

public:
    void onDrawing();
    void finishDrawing();

    void onAddPoint(RealVector2D const& worldPos);
    void onRemoveLastPoint();

private:
    CreatorWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void processBackground() override;
    bool isShown() override;

    void processToolbar();

    void processCreateObject();
    void processCreateRectangle();
    void processCreateHexagon();
    void processCreateDisc();
    void processDrawing();
    void processCreateLine();
    void processCreateCurve();
    void processCreatePolygon();

    void updateInteractionMode();

    bool beginParameterPanel();
    void endParameterPanel();
    void processColorWidget();
    void processMaterialWidgets();
    void processObjectDistanceWidget();
    void processStickyWidget();
    void processStaticWidget();
    bool processBuildButton();
    bool processPointButtons(int minNumPoints);
    void abortPointPlacement();

    void processPointPreview(std::vector<RealVector2D> const& path, bool closed) const;
    void processControlPolygonPreview() const;

    void createSingleObject();
    void createRectangle();
    void createHexagon();
    void createDisc();
    void createObjectNetwork(std::vector<RealVector2D> const& positions, float connectionDistance);

    std::vector<RealVector2D> calcBezierCurvePath() const;

    ContentDesc convertToEnergyParticles(ContentDesc const& description) const;

    void validateAndCorrect();
    bool isEnergyMaterial() const;
    bool isPointPlacementMode() const;
    InteractionMode getInteractionMode() const;

    ObjectTypeDesc getObjectTypeDesc() const;

    RealVector2D getRandomPos() const;

    float _energy = 100.0f;
    float _stiffness = 1.0f;
    bool _static = false;
    float _objectDistance = 1.0f;
    float _glow = 0.0f;
    bool _makeSticky = false;

    // Rectangle data
    int _rectHorizontalObjects = 10;
    int _rectVerticalObjects = 10;

    // Hexagon data
    int _layers = 10;

    // Disc data
    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    // Drawing data
    CreationMaterial _material = CreationMaterial_Solid;
    ContentDesc _drawingDescription;
    DescEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    std::vector<RealVector2D> _points;
    bool _pointPlacementActive = false;

    CreationMode _mode = CreationMode_Drawing;
};
