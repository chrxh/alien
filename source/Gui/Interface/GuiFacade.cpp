#include "GuiFacade.h"

GuiFacade _GuiFacade::_instance;

GuiFacade _GuiFacade::get()
{
    return _instance;
}
