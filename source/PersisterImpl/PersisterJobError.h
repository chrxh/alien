#pragma once

#include "PersisterInterface/PersisterController.h"

class _PersisterJobError
{
public:
    _PersisterJobError(PersisterJobId const& id, bool critical, PersisterErrorInfo const& errorInfo);
    virtual ~_PersisterJobError() = default;

    PersisterJobId const& getId() const;
    bool isCritical() const;
    PersisterErrorInfo const& getErrorInfo() const;

protected:
    PersisterJobId _id;
    bool _critical = true;
    PersisterErrorInfo _errorInfo;
};
using PersisterJobError = std::shared_ptr<_PersisterJobError>;
