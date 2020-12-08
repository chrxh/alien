#pragma once

#include "Job.h"

Job::Job(string id, QObject* parent)
    : QObject(parent), _id(id) 
{
}

string const& Job::getId() const
{
    return _id;
}