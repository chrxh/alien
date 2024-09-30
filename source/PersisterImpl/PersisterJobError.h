#pragma once

#include "PersisterInterface/PersisterController.h"

class _PersisterJobError
{
public:
    PersisterJobId getId() const;

    _PersisterJobError(PersisterJobId const& id, PersisterErrorInfo const& errorInfo);
    virtual ~_PersisterJobError() = default;

    PersisterErrorInfo const& getPersisterErrorInfo() const;

protected:
    PersisterJobId _id;
    PersisterErrorInfo _errorInfo;
};
using PersisterJobError = std::shared_ptr<_PersisterJobError>;
