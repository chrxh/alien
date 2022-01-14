#pragma once

#include "EngineInterface/SimulationController.h"

#include "Definitions.h"

enum class CreationMode
{
    CreateParticle,
    CreateCell,
    CreateRect,
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

    RealVector2D getRandomPos() const;

    bool _on = false;

    float _energy = 100.0f;
    float _distance = 1.0f;
    int _maxConnections = 1;
    bool _increaseBranchNumber = true;
    int _lastBranchNumber = 0;

    CreationMode _mode = CreationMode::CreateCell;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
};
