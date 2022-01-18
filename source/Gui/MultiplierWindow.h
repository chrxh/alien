#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"

enum class MultiplierMode
{
    Grid,
    Random
};

class _MultiplierWindow : public _AlienWindow
{
public:
    _MultiplierWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport);

private:
    void processIntern() override;
    void processGridPanel();

    EditorModel _editorModel; 
    SimulationController _simController;
    Viewport _viewport;

    MultiplierMode _mode = MultiplierMode::Grid;

    int _horizontalNumber = 10;
    float _horizontalDistance = 10.0f;
    float _horizontalAngleInc = 0;
    float _horizontalVelXinc = 0;
    float _horizontalVelYinc = 0;
    float _horizontalAngularVelInc = 0;

    int _verticalNumber = 10;
    float _verticalDistance = 10.0f;
    float _verticalAngleInc = 0;
    float _verticalVelXinc = 0;
    float _verticalVelYinc = 0;
    float _verticalAngularVelInc = 0;
};