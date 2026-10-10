#include "MainLoopEntity.h"

#include "MainLoopEntityController.h"

void MainLoopEntity::registerObject()
{
    MainLoopEntityController::get().registerObject(this);
}

void MainLoopEntity::setup()
{
    init();
    registerObject();
}
