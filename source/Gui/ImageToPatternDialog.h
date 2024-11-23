#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class ImageToPatternDialog : public MainLoopEntity<SimulationFacade>
{
    MAKE_SINGLETON(ImageToPatternDialog);

public:
	void show();

private:
    void init(SimulationFacade simulationFacade) override;
    void shutdown() override;
    void process() override {}

    SimulationFacade _simulationFacade;

    std::string _startingPath;
};