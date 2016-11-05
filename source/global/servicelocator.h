#ifndef SERVICELOCATOR_H
#define SERVICELOCATOR_H

#include <QString>
#include <QMap>

class ServiceLocator
{
public:
    static ServiceLocator& getInstance ();

    template< typename T >
    void registerService (const QString& name, T* service);

    template< typename T >
    T* getService (const QString& name);

private:
    ServiceLocator () {}

public:
    ServiceLocator (ServiceLocator const&) = delete;
    void operator= (ServiceLocator const&) = delete;

private:
    QMap< QString, void* > _services;
};


//implementations
template< typename T >
void ServiceLocator::registerService (const QString& name, T* service)
{
    _services[name] = service;
}

template< typename T >
T* ServiceLocator::getService (const QString& name)
{
    if( _services.contains(name) )
        return static_cast< T* >(_services[name]);
    else
        return static_cast< T* >(0);
}

#endif // SERVICELOCATOR_H
