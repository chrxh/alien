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

template <typename... Dependencies>
class MainLoopEntity : public AbstractMainLoopEntity
{
public:
    virtual ~MainLoopEntity() override = default;

    void setup(Dependencies... dependencies);

protected:
    virtual void init(Dependencies... dependencies) = 0;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

template <typename ... Dependencies>
void MainLoopEntity<Dependencies...>::setup(Dependencies... dependencies)
{
    init(dependencies...);
    registerObject();
}
