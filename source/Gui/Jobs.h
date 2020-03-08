#pragma once

#include "Definitions.h"

class _Job
{
public:
    virtual ~_Job() = default;
    using ExecutionFunc = std::function<void(Serializer*)>;
    _Job(ExecutionFunc const& executionFunc)
        : _executionFunc(executionFunc)
    {}

    ExecutionFunc getExecutionFunction() const
    {
        return _executionFunc;
    }

private:
    std::function<void(Serializer*)> _executionFunc;
};
using Job = shared_ptr<_Job>;
