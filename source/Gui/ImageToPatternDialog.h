#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class ImageToPatternDialog : public MainLoopEntity
{
    MAKE_SINGLETON(ImageToPatternDialog);

public:

	void init(SimulationFacade const& simulationFacade);
    
	void show();

private:
    void shutdown() override;
    void process() override {}

    SimulationFacade _simulationFacade;

    std::string _startingPath;
};