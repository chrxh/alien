#pragma once

class MainLoopEntity
{
public:
    virtual ~MainLoopEntity() = default;

    virtual void shutdown() = 0;
    virtual void process() = 0;
};
