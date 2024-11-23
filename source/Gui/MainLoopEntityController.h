#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "MainLoopEntity.h"


class MainLoopEntityController
{
    MAKE_SINGLETON(MainLoopEntityController);

public:
    void registerObject(AbstractMainLoopEntity* object);

    void shutdown();
    void process();

private:
    std::vector<AbstractMainLoopEntity*> _objects;
};
