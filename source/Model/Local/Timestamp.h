#ifndef TIMESTAMP_H
#define TIMESTAMP_H

#include "Model/Api/Definitions.h"
#include "Model/Local/UnitContext.h"

class Timestamp
{
public:
	inline Timestamp(UnitContext* context);
	virtual ~Timestamp() = default;

	virtual void setContext(UnitContext * context);
	inline void incTimestampIfFit();

protected:
	inline bool isTimestampFitting() const;

	UnitContext* _context = nullptr;

private:
	uint64_t _timestamp = 0;
};

/********************* inline methods ******************/
Timestamp::Timestamp(UnitContext* context)
{
	_context = context;
	_timestamp = _context->getTimestamp();
}

void Timestamp::incTimestampIfFit()
{
	if (isTimestampFitting()) {
		++_timestamp;
	}
}

bool Timestamp::isTimestampFitting() const
{
	return _timestamp == _context->getTimestamp();
}



#endif // TIMESTAMP_H
