#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "ShutdownInterface.h"


class ShutdownController
{
    MAKE_SINGLETON(ShutdownController);

public:
    void registerObject(ShutdownInterface* object);

    void shutdown();

private:
    std::vector<ShutdownInterface*> _objects;
};
