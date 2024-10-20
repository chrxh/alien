#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "MainLoopEntity.h"


class MainLoopEntityController
{
    MAKE_SINGLETON(MainLoopEntityController);

public:
    void registerObject(MainLoopEntity* object);

    void shutdown();
    void process();

private:
    std::vector<MainLoopEntity*> _objects;
};
