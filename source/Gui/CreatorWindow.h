#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionEditService.h"

#include "Definitions.h"
#include "AlienWindow.h"

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

class _CreatorWindow : public _AlienWindow
{
public:
    _CreatorWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport);

    void onDrawing();
    void finishDrawing();

private:
    void processIntern();

    void createCell();
    void createParticle();
    void createRectangle();
    void createHexagon();
    void createDisc();

    void validationAndCorrection();

    RealVector2D getRandomPos() const;
    void incExecutionNumber();

    float _energy = 100.0f;
    float _stiffness = 1.0f;
    bool _barrier = false;
    float _cellDistance = 1.0f;
    bool _makeSticky = false;
    int _maxConnections = 6;
    bool _ascendingExecutionNumbers = true;
    int _lastExecutionNumber = 0;

    //rectangle
    int _rectHorizontalCells = 10;
    int _rectVerticalCells = 10;

    //hexagon
    int _layers = 10;

    //disc
    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    //drawing
    DataDescription _drawing;
    DescriptionEditService::Occupancy _drawingOccupancy;
    RealVector2D _lastDrawPos;

    CreationMode _mode = CreationMode_Drawing;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
};
