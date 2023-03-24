#pragma once

#include "Definitions.h"

class DelayedExecutionController
{
public:
    static DelayedExecutionController& getInstance();

    void process();

    void executeLater(std::function<void(void)> const& execFunc);

private:
    std::function<void(void)> _execFunc;
    int _timer = 0;
};

inline void delayedExecution(std::function<void(void)> const& execFunc)
{
    DelayedExecutionController::getInstance().executeLater(execFunc);
}