#pragma once

class AbstractMainLoopEntity
{
public:
    virtual ~AbstractMainLoopEntity() = default;

    virtual void process() = 0;
    virtual void shutdown() = 0;

protected:
    void registerObject();
};

class MainLoopEntity : public AbstractMainLoopEntity
{
public:
    virtual ~MainLoopEntity() override = default;

    void setup();

protected:
    virtual void init() = 0;
};
