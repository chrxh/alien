#pragma once

#include <functional>

class ExitScopeGuard
{
public:
    ExitScopeGuard(std::function<void()> func)
        : _func(func)
    {}
    ~ExitScopeGuard() { _func(); }

private:
    std::function<void()> _func;
};
