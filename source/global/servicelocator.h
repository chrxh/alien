#ifndef SERVICELOCATOR_H
#define SERVICELOCATOR_H

#include <QString>
#include <QMap>
#include <typeinfo>

class ServiceLocator
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
    QMap< size_t, void* > _services;
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
    size_t hashCode = typeid(T).hash_code();
    if( _services.contains(hashCode) )
        return static_cast< T* >(_services[hashCode]);
    else
        return static_cast< T* >(0);
}

#endif // SERVICELOCATOR_H
