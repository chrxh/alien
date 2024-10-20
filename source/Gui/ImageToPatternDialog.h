#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "ShutdownInterface.h"

class ImageToPatternDialog : public ShutdownInterface
{
    MAKE_SINGLETON(ImageToPatternDialog);

public:

	void init(SimulationFacade const& simulationFacade);
    
	void show();

private:
    void shutdown() override;

    SimulationFacade _simulationFacade;

    std::string _startingPath;
};