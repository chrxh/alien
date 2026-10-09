#include "RenderingFacade.h"

RenderingFacade _RenderingFacade::_instance;

RenderingFacade _RenderingFacade::get()
{
    return _instance;
}
