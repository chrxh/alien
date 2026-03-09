#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>

#include "AlienWindow.h"
#include "Definitions.h"

using CreationMode = int;
enum CreationMode_
{
    CreationMode_CreateParticle,
    CreationMode_CreateObject,
    CreationMode_CreateRectangle,
    CreationMode_CreateHexagon,
    CreationMode_CreateDisc,
    CreationMode_Drawing
};

using DrawingType = int;
enum DrawingType_
{
    DrawingType_Solid,
    DrawingType_Fluid,
    DrawingType_Count
};

class CreatorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatorWindow);

public:
    void onDrawing();
    void finishDrawing();

private:
    CreatorWindow();

    void initIntern() override;
    void processIntern() override;
    bool isShown() override;

    void createCell();
    void createParticle();
    void createRectangle();
    void createHexagon();
    void createDisc();

    void validateAndCorrect();

    RealVector2D getRandomPos() const;

    float _energy = 100.0f;
    float _stiffness = 1.0f;
    bool _fixed = false;
    float _cellDistance = 1.0f;
    bool _makeSticky = false;

    //rectangle
    int _rectHorizontalCells = 10;
    int _rectVerticalCells = 10;

    //hexagon
    int _layers = 10;

    //disc
    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    //drawing
    DrawingType _drawingType = DrawingType_Solid;
    Desc _drawingDescription;
    DescEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    CreationMode _mode = CreationMode_Drawing;
};
