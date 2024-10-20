#pragma once

class ShutdownInterface
{
public:
    virtual ~ShutdownInterface() = default;

    virtual void shutdown() = 0;
};
