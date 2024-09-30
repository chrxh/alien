#include "PersisterJobError.h"

_PersisterJobError::_PersisterJobError(PersisterJobId const& id, PersisterErrorInfo const& errorInfo)
    : _id(id), _errorInfo(errorInfo)
{
}

PersisterJobId const& _PersisterJobError::getId() const
{
    return _id;
}

PersisterErrorInfo const& _PersisterJobError::getErrorInfo() const
{
    return _errorInfo;
}
