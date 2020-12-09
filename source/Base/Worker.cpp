#include "Worker.h"

bool _Worker::contains(string const& id)
{
    return _jobById.find(id) != _jobById.end();
}

bool _Worker::add(Job* job)
{
    if (_jobById.find(job->getId()) != _jobById.end()) {
        return false;
    }

    _jobs.emplace_back(job);
    _jobById.emplace(job->getId(), job);

    return true;
}

void _Worker::process()
{
    vector<Job*> newJobQueue;
    newJobQueue.reserve(_jobs.size());

    bool blockingJobs = false;
    for (auto const& job : _jobs) {
        if (blockingJobs) {
            newJobQueue.emplace_back(job);
            continue;
        }
        job->process();
        if (!job->isFinished()) {
            newJobQueue.emplace_back(job);
        }
        if (job->isBlocking()) {
            blockingJobs = true;
        }
        if (job->isFinished()) {
            _jobById.erase(job->getId());
            delete job;
        }
    }
    _jobs = newJobQueue;
}

