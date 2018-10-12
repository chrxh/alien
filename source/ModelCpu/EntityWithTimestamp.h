#pragma once

#include "ModelBasic/Definitions.h"
#include "UnitContext.h"

class EntityWithTimestamp
{
public:
	inline EntityWithTimestamp(UnitContext* context);
	virtual ~EntityWithTimestamp() = default;

	void setContext(UnitContext * context);
	UnitContext* getContext() const;
	inline void incTimestampIfFit();

protected:
	inline bool isTimestampFitting() const;

	UnitContext* _context = nullptr;

private:
	uint64_t _timestamp = 0;
};

/********************* inline methods ******************/
EntityWithTimestamp::EntityWithTimestamp(UnitContext* context)
{
	_context = context;
	_timestamp = _context->getTimestamp();
}

void EntityWithTimestamp::incTimestampIfFit()
{
	if (isTimestampFitting()) {
		++_timestamp;
	}
}

bool EntityWithTimestamp::isTimestampFitting() const
{
	return _timestamp == _context->getTimestamp();
}

