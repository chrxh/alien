#pragma once

#include "Base/Singleton.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class DelayedExecutionController : public MainLoopEntity
{
    MAKE_SINGLETON(DelayedExecutionController);

public:
    void init();

    void executeLater(std::function<void(void)> const& execFunc);

private:
    void process() override;
    void shutdown() override {}

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