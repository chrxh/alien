#pragma once

#include <functional>

class OutOfScopeGuard
{
public:
    OutOfScopeGuard(std::function<void()> func)
        : _func(func)
    {}
    ~OutOfScopeGuard() { _func(); }

private:
    std::function<void()> _func;
};
