#pragma once

#include "EngineInterface/SimulationController.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

enum class CreationMode
{
    CreateParticle,
    CreateCell,
    CreateRectangle,
    CreateHexagon,
    CreateDisc,
    Drawing
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
    void drawing();

    RealVector2D getRandomPos() const;
    void incBranchNumber();

    float _energy = 100.0f;
    bool _barrier = false;
    float _cellDistance = 1.0f;
    bool _makeSticky = false;
    int _maxConnections = 6;
    bool _ascendingBranchNumbers = true;
    int _lastBranchNumber = 0;

    //rectangle
    int _rectHorizontalCells = 10;
    int _rectVerticalCells = 10;

    //hexagon
    int _layers = 10;

    //disc
    float _outerRadius = 10.0f;
    float _innerRadius = 5.0f;

    //paint
    DataDescription _drawing;

    CreationMode _mode = CreationMode::CreateCell;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
};
