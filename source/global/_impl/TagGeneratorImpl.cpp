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
	std::lock_guard<std::mutex> lock(_mutex);
	return ++_tag;
}

quint64 TagGeneratorImpl::getCurrentTag()
{
	std::lock_guard<std::mutex> lock(_mutex);
	return _tag;
}

void TagGeneratorImpl::setSeed(quint64 tag)
{
	std::lock_guard<std::mutex> lock(_mutex);
	_tag = tag;
}
