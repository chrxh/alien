#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class ImageToPatternDialog : public MainLoopEntity
{
    MAKE_SINGLETON(ImageToPatternDialog);

public:
    void show();

private:
    void init() override;
    void shutdown() override;
    void process() override {}

    std::string _startingPath;
};
