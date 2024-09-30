#pragma once

#include "PersisterInterface/PersisterController.h"

class _PersisterJobError
{
public:
    _PersisterJobError(PersisterJobId const& id, PersisterErrorInfo const& errorInfo);
    virtual ~_PersisterJobError() = default;

    PersisterJobId const& getId() const;
    PersisterErrorInfo const& getErrorInfo() const;

protected:
    PersisterJobId _id;
    PersisterErrorInfo _errorInfo;
};
using PersisterJobError = std::shared_ptr<_PersisterJobError>;
