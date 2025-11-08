#include "PersisterFacadeRegistration.h"

#include "PersisterFacadeImpl.h"

PersisterFacadeRegistration::PersisterFacadeRegistration()
{
    _PersisterFacade::set(std::make_shared<_PersisterFacadeImpl>());
}

// Global variable to trigger registration
static PersisterFacadeRegistration persisterFacadeRegistration;
