#pragma once

#include <PersisterInterface/PersisterFacade.h>

class PersisterFacadeProvider
{
public:
    static void setPersisterFacade(PersisterFacade const& persisterFacade);
    static PersisterFacade getPersisterFacade();

private:
    static PersisterFacade _persisterFacade;
};
