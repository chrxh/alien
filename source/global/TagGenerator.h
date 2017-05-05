#ifndef TAGGENERATOR_H
#define TAGGENERATOR_H

#include "Definitions.h"

class TagGenerator
{
public:
	virtual ~TagGenerator() = default;

	virtual quint64 getNewTag() = 0;
	virtual quint64 getCurrentTag() = 0;
	virtual void setSeed(quint64 tag) = 0;
};

#endif // TAGGENERATOR_H
