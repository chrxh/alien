#ifndef TAGGENERATORIMPL_H
#define TAGGENERATORIMPL_H

#include <mutex>

#include "global/TagGenerator.h"

class TagGeneratorImpl
	: public TagGenerator
{
public:
	TagGeneratorImpl();
	virtual ~TagGeneratorImpl() = default;

	virtual quint64 getNewTag() override;
	virtual quint64 getCurrentTag() override;
	virtual void setSeed(quint64 tag) override;

private:
	std::mutex _mutex;
	quint64 _tag = 0;
};

#endif // TAGGENERATORIMPL_H
