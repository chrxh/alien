#pragma once

class MainLoopEntity
{
public:
    virtual ~MainLoopEntity() = default;

    virtual void process() = 0;
    virtual void shutdown() = 0;
};
