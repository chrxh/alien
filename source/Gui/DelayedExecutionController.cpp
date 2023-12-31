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
    std::vector<ExecutionData> delayedExecDatas;
    std::vector<ExecutionData> toExecute;
    for (auto& execData : _execDatas) {
        if (--execData.timer == 0) {
            toExecute.emplace_back(execData);
        } else {
            delayedExecDatas.emplace_back(execData);
        }
    }
    _execDatas = delayedExecDatas;

    for (auto& execData : toExecute) {
        execData.func();
    }
}

void DelayedExecutionController::executeLater(std::function<void()> const& execFunc)
{
    _execDatas.emplace_back(execFunc, 2);
}
