#pragma once

#include "EngineInterface/SimulationController.h"

#include "Definitions.h"

enum class CreationMode
{
    CreateParticle,
    CreateCell,
    CreateRectangle,
    CreateHexagon,
    CreateDisc
};

class _CreatorWindow
{
public:
    _CreatorWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport);
    ~_CreatorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void createCell();
    void createParticle();
    void createRectangle();

    RealVector2D getRandomPos() const;

    bool _on = false;

    float _energy = 100.0f;
    float _cellDistance = 1.0f;
    bool _autoMaxConnections = true;
    int _maxConnections = 1;
    bool _increaseBranchNumber = true;
    int _lastBranchNumber = 0;

    //rectangle
    int _rectHorizontalCells = 10;
    int _rectVerticalCells = 10;

    CreationMode _mode = CreationMode::CreateCell;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
};
