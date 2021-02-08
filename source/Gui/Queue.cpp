#include "EngineInterface/Serializer.h"

#include "Queue.h"

Queue::Queue(QObject * parent)
    : QObject(parent)
{
}

void Queue::init(Serializer * serializer)
{
    _serializer = serializer;

    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();

    _connections.push_back(
        connect(_serializer, &Serializer::serializationFinished, this, &Queue::processingJobs));
}

void Queue::add(ExecuteLaterFunc const & job)
{
    _jobs.emplace_back(job);
}

void Queue::processingJobs()
{
    for (auto const& job : _jobs) {
        auto const executionJob = job->getExecutionFunction();
        executionJob(_serializer);
    }
    _jobs.clear();
}
