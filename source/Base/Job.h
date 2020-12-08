#pragma once

#include "Definitions.h"

class BASE_EXPORT Job : public QObject
{
    Q_OBJECT
public:
    Job(string id, QObject* parent);
    virtual ~Job() = default;

    string const& getId() const;

    virtual void process() = 0;
    virtual bool isFinished() const = 0;
    virtual bool isBlocking() const = 0;

private:
    string _id;
};
