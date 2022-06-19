#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _ImageToPatternDialog
{
public:

	_ImageToPatternDialog(Viewport const& viewport, SimulationController const& simController);
    ~_ImageToPatternDialog();

	void show();

private:
    Viewport _viewport;
    SimulationController _simController;

    std::string _startingPath;
};