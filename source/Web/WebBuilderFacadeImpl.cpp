#include "WebBuilderFacadeImpl.h"
#include "WebControllerImpl.h"

WebController * WebBuilderFacadeImpl::buildWebController() const
{
    return new WebControllerImpl();
}
