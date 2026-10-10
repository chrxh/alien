#pragma once

class MainLoopEntity
{
public:
    virtual ~MainLoopEntity() = default;

    void setup();
    virtual void process() = 0;
    virtual void shutdown() = 0;

protected:
    void registerObject();
    virtual void init() = 0;
};
