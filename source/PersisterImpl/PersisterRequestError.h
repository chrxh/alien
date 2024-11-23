#pragma once

#include "PersisterInterface/PersisterFacade.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "PersisterInterface/SenderId.h"

class _PersisterRequestError
{
public:
    _PersisterRequestError(PersisterRequestId const& requestId, SenderId const& senderId, PersisterErrorInfo const& errorInfo);
    virtual ~_PersisterRequestError() = default;

    PersisterRequestId const& getRequestId() const;
    SenderId const& getSenderId() const;
    PersisterErrorInfo const& getErrorInfo() const;

protected:
    PersisterRequestId _requestId;
    SenderId _senderId;
    PersisterErrorInfo _errorInfo;
};
using PersisterRequestError = std::shared_ptr<_PersisterRequestError>;
