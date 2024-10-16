#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class _ImageToPatternDialog
{
public:

	_ImageToPatternDialog(SimulationFacade const& simulationFacade);
    ~_ImageToPatternDialog();

	void show();

private:
    SimulationFacade _simulationFacade;

    std::string _startingPath;
};