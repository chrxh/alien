#include "PersisterJobError.h"

_PersisterJobError::_PersisterJobError(PersisterJobId const& id, bool critical, PersisterErrorInfo const& errorInfo)
    : _id(id)
    , _critical(critical)
    , _errorInfo(errorInfo)
{
}

PersisterJobId const& _PersisterJobError::getId() const
{
    return _id;
}

bool _PersisterJobError::isCritical() const
{
    return _critical;
}

PersisterErrorInfo const& _PersisterJobError::getErrorInfo() const
{
    return _errorInfo;
}
