#include "PersisterFacade.h"

PersisterFacade _PersisterFacade::_instance;

PersisterFacade _PersisterFacade::get()
{
    return _instance;
}
