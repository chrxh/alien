#pragma once

#include <cstddef>
#include <typeinfo>

#include "DllExport.h"

struct ServiceLocatorImpl;

class BASE_EXPORT ServiceLocator
{
public:
    static ServiceLocator& getInstance ();

    template< typename T >
    void registerService (T* service);

    template< typename T >
    T* getService ();

public:
    ServiceLocator (ServiceLocator const&) = delete;
    void operator= (ServiceLocator const&) = delete;

private:
    ServiceLocator();
    ~ServiceLocator();

    ServiceLocatorImpl* _pimpl;

    void lock();
    void unlock();
    void* findServiceImpl(size_t hashCode) const;
    void registerServiceImpl(size_t hashCode, void* service);
};


//implementations
template< typename T >
void ServiceLocator::registerService (T* service)
{
    registerServiceImpl(typeid(T).hash_code(), service);
}

template< typename T >
T* ServiceLocator::getService ()
{
    lock();
    size_t hashCode = typeid(T).hash_code();
    auto result = findServiceImpl(hashCode);
    unlock();
    return static_cast<T*>(result);
}
