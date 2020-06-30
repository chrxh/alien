#pragma once
#include <QObject>

#include "Jobs.h"

class Worker
    : public QObject
{
    Q_OBJECT
public:
    Worker(QObject * parent = nullptr);
    virtual ~Worker() = default;

    void init(Serializer* serializer);

    void addJob(Job const& job);

private:
    Q_SLOT void processingJobs();

private:
    list<QMetaObject::Connection> _connections;

    Serializer* _serializer;
    list<Job> _jobs;
};
