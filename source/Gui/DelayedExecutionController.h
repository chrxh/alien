#pragma once

#include "Base/Singleton.h"
#include "Definitions.h"

class DelayedExecutionController
{
    MAKE_SINGLETON(DelayedExecutionController);

public:
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
    DelayedExecutionController::get().executeLater(execFunc);
}