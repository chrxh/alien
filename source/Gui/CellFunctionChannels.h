#pragma once

#include <string>
#include <vector>

#include <EngineInterface/Descs.h>
#include <EngineInterface/GenomeDesc.h>

// A channel of the outgoing signal that a cell function reads and/or overwrites after the neural net has been evaluated
struct CellFunctionChannel
{
    int channel = 0;
    std::string readLabel;   // Empty if the channel is not read
    std::string writeLabel;  // Empty if the channel is not overwritten
};

// A cell function accessing the outgoing signal. A cell can have several of them, e.g. a sensor with a constructor.
struct CellFunctionModule
{
    std::string name;
    std::vector<CellFunctionChannel> channels;  // Ordered by channel
};

class CellFunctionChannels
{
public:
    static std::vector<CellFunctionModule> getModules(NodeDesc const& node);
    static std::vector<CellFunctionModule> getModules(CellDesc const& cell);
};
