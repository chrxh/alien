#include "PersisterRequestError.h"

_PersisterRequestError::_PersisterRequestError(PersisterRequestId const& id, SenderId const& senderId, PersisterErrorInfo const& errorInfo)
    : _requestId(id)
    , _senderId(senderId)
    , _errorInfo(errorInfo)
{}

PersisterRequestId const& _PersisterRequestError::getRequestId() const
{
    return _requestId;
}

SenderId const& _PersisterRequestError::getSenderId() const
{
    return _senderId;
}

PersisterErrorInfo const& _PersisterRequestError::getErrorInfo() const
{
    return _errorInfo;
}
