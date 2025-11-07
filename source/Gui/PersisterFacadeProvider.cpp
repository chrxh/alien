#include "PersisterFacadeProvider.h"

PersisterFacade PersisterFacadeProvider::_persisterFacade;

void PersisterFacadeProvider::setPersisterFacade(PersisterFacade const& persisterFacade)
{
    _persisterFacade = persisterFacade;
}

PersisterFacade PersisterFacadeProvider::getPersisterFacade()
{
    return _persisterFacade;
}
