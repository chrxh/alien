#include "ModelBasic/Serializer.h"

#include "Worker.h"

Worker::Worker(QObject * parent)
    : QObject(parent)
{
}

void Worker::init(Serializer * serializer)
{
    _serializer = serializer;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();

    _connections.push_back(
        connect(_serializer, &Serializer::serializationFinished, this, &Worker::processingJobs));
}

void Worker::addJob(Job const & job)
{
    _jobs.emplace_back(job);
}

void Worker::processingJobs()
{
    for (auto const& job : _jobs) {
        auto const executionJob = job->getExecutionFunction();
        executionJob(_serializer);
    }
    _jobs.clear();
}
