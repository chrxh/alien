#include "global/ServiceLocator.h"

#include "TagGeneratorImpl.h"


namespace {
	TagGeneratorImpl instance;
}

TagGeneratorImpl::TagGeneratorImpl()
{
	ServiceLocator::getInstance().registerService<TagGenerator>(this);
}

quint64 TagGeneratorImpl::getNewTag()
{
	return _tag.fetch_add(1);
}

quint64 TagGeneratorImpl::getCurrentTag()
{
	return _tag.load();
}

void TagGeneratorImpl::setSeed(quint64 tag)
{
	_tag.store(tag);
}
