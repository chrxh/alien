#pragma once

#include "Definitions.h"

class _ExecuteLaterFunc
{
public:
    virtual ~_ExecuteLaterFunc() = default;
    using ExecutionFunc = std::function<void(Serializer*)>;
    _ExecuteLaterFunc(ExecutionFunc const& executionFunc)
        : _executionFunc(executionFunc)
    {}

    ExecutionFunc getExecutionFunction() const
    {
        return _executionFunc;
    }

private:
    std::function<void(Serializer*)> _executionFunc;
};
using ExecuteLaterFunc = shared_ptr<_ExecuteLaterFunc>;
