#pragma once
#include <QObject>

#include "Jobs.h"

class Queue
    : public QObject
{
    Q_OBJECT
public:
    Queue(QObject * parent = nullptr);
    virtual ~Queue() = default;

    void init(Serializer* serializer);

    void add(ExecuteLaterFunc const& job);

private:
    Q_SLOT void processingJobs();

private:
    list<QMetaObject::Connection> _connections;

    Serializer* _serializer;
    list<ExecuteLaterFunc> _jobs;
};
