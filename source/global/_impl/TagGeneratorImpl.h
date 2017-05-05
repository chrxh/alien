#ifndef TAGGENERATORIMPL_H
#define TAGGENERATORIMPL_H

#include <atomic>

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
	std::atomic<quint64> _tag = 0;
};

#endif // TAGGENERATORIMPL_H
