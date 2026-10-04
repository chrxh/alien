#pragma once

// Identifies a domain of the domain decomposition: the world is split into vertical strips and each strip is simulated
// by its own SimulationData, possibly on its own GPU
struct DomainContext
{
    int index = 0;
    int numDomains = 1;
};
