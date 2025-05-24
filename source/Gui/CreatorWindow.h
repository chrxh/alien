#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionEditService.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "Base/Singleton.h"

using CreationMode = int;
enum CreationMode_
{
    CreationMode_CreateParticle,
    CreationMode_CreateCell,
    CreationMode_CreateRectangle,
    CreationMode_CreateHexagon,
    CreationMode_CreateDisc,
    CreationMode_Drawing
};

class CreatorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatorWindow);

public:
    void onDrawing();
    void finishDrawing();

private:
    CreatorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
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
    bool _barrier = false;
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
    CollectionDescription _drawingDescription;
    DescriptionEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    CreationMode _mode = CreationMode_Drawing;

    SimulationFacade _simulationFacade;
};
