#include "ShutdownController.h"

#include <ranges>
void ShutdownController::registerObject(ShutdownInterface* object)
{
    _objects.emplace_back(object);
}

void ShutdownController::shutdown()
{
    for (auto const& object : _objects | std::views::reverse) {
        object->shutdown();
    }
}