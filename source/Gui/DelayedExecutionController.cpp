#include <imgui.h>

#include "AlienImGui.h"
#include "DelayedExecutionController.h"


DelayedExecutionController& DelayedExecutionController::getInstance()
{
    static DelayedExecutionController instance;
    return instance;
}

void DelayedExecutionController::process()
{
    if (_timer == 0) {
        return;
    }
    --_timer;
    if (_timer == 0) {
        _execFunc();
    }
}

void DelayedExecutionController::executeLater(std::function<void()> const& execFunc)
{
    _execFunc = execFunc;
    _timer = 2;
}
