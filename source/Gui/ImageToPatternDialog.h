#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _ImageToPatternDialog
{
public:

	_ImageToPatternDialog(SimulationController const& simController);
    ~_ImageToPatternDialog();

	void show();

private:
    SimulationController _simController;

    std::string _startingPath;
};