#include "MainLoopEntityController.h"

#include <ranges>

void MainLoopEntityController::registerObject(AbstractMainLoopEntity* object)
{
    _objects.emplace_back(object);
}

void MainLoopEntityController::shutdown()
{
    for (auto const& object : _objects | std::views::reverse) {
        object->shutdown();
    }
}

void MainLoopEntityController::process()
{
    for (auto const& object : _objects | std::views::reverse) {
        object->process();
    }
}