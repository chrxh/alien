#include "WebBuilderFacadeImpl.h"
#include "WebAccessImpl.h"

WebAccess * WebBuilderFacadeImpl::buildWebController() const
{
    return new WebAccessImpl();
}
