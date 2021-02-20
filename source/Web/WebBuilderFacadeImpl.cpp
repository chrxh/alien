#include "WebBuilderFacadeImpl.h"
#include "WebAccessImpl.h"

WebAccess * WebBuilderFacadeImpl::buildWebAccess() const
{
    return new WebAccessImpl();
}
