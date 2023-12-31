#pragma once

#include "Definitions.h"

class DelayedExecutionController
{
public:
    static DelayedExecutionController& getInstance();

    void process();

    void executeLater(std::function<void(void)> const& execFunc);

private:
    struct ExecutionData
    {
        std::function<void(void)> func;
        int timer = 0;
    };
    std::vector<ExecutionData> _execDatas;
};

inline void delayedExecution(std::function<void(void)> const& execFunc)
{
    DelayedExecutionController::getInstance().executeLater(execFunc);
}