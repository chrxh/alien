#pragma 

#include "Job.h"
#include "Definitions.h"

class BASE_EXPORT _Worker
{
public:
    bool contains(string const& id);

    //returns false if job is already in queue
    bool add(Job* job);

    /*
    Job get(string const& id) const
    {
    auto findResult = _jobById.find(id);
    if (findResult != _jobById.end()) {
    return findResult->second;
    }
    return nullptr;
    }

    Job getFirst() const
    {
    if (_jobs.empty()) {
    return nullptr;
    }
    return _jobs.front();
    }
    */

    void process();
private:
    vector<Job*> _jobs;
    unordered_map<string, Job*> _jobById;
};

