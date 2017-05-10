#ifndef SERVICELOCATOR_H
#define SERVICELOCATOR_H

#include <QString>
#include <mutex>
#include <typeinfo>
#include "Definitions.h"

class BASE_EXPORT ServiceLocator
{
public:
    static ServiceLocator& getInstance ();

    template< typename T >
    void registerService (T* service);

    template< typename T >
    T* getService ();

private:
    ServiceLocator () {}
	~ServiceLocator() {}

public:
    ServiceLocator (ServiceLocator const&) = delete;
    void operator= (ServiceLocator const&) = delete;

private:
    map< size_t, void* > _services;
	std::mutex _mutex;
};


//implementations
template< typename T >
void ServiceLocator::registerService (T* service)
{
    _services[typeid(T).hash_code()] = service;
}

template< typename T >
T* ServiceLocator::getService ()
{
	std::lock_guard<std::mutex> lock(_mutex);
    size_t hashCode = typeid(T).hash_code();
	auto serviceIter = _services.find(hashCode);
	if (serviceIter != _services.end()) {
		return static_cast<T*>(serviceIter->second);
	}
    return static_cast<T*>(nullptr);
}

#endif // SERVICELOCATOR_H
