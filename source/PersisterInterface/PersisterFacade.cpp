#include "PersisterFacade.h"

PersisterFacade _PersisterFacade::_instance;

void _PersisterFacade::set(PersisterFacade const& instance)
{
    _instance = instance;
}

PersisterFacade _PersisterFacade::get()
{
    return _instance;
}
