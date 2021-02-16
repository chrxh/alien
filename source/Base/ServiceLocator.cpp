#include <mutex>
#include <map>

#include "ServiceLocator.h"

struct ServiceLocatorImpl
{
    std::map<size_t, void*> services;
    std::mutex mutex;
};

ServiceLocator& ServiceLocator::getInstance ()
{
    static ServiceLocator instance;
    return instance;
}

ServiceLocator::ServiceLocator()
{
    _pimpl = new ServiceLocatorImpl();
}

ServiceLocator::~ServiceLocator()
{
    delete _pimpl;
}

void ServiceLocator::lock()
{
    _pimpl->mutex.lock();
}

void ServiceLocator::unlock()
{
    _pimpl->mutex.unlock();
}

void* ServiceLocator::findServiceImpl(size_t hashCode) const
{
    auto serviceIter = _pimpl->services.find(hashCode);
    if (serviceIter != _pimpl->services.end()) {
        return serviceIter->second;
    }
    return nullptr;
}

void ServiceLocator::registerServiceImpl(size_t hashCode, void* service)
{
    _pimpl->services[hashCode] = service;
}
