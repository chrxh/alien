#include "EntityWithTimestamp.h"

void EntityWithTimestamp::setContext(UnitContext * context)
{
	_context = context;
}

UnitContext* EntityWithTimestamp::getContext() const
{
	return _context;
}
