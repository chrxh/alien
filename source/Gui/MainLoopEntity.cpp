#include "MainLoopEntity.h"

#include "MainLoopEntityController.h"

void AbstractMainLoopEntity::registerObject()
{
    MainLoopEntityController::get().registerObject(this);
}

void MainLoopEntity::setup()
{
    init();
    registerObject();
}
