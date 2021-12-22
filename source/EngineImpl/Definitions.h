#pragma once

#include <boost/shared_ptr.hpp>

struct DataRolloutDescription;
struct CellRolloutDescription;
struct ParticleRolloutDescription;

class _SimulationController;
using SimulationController = boost::shared_ptr<_SimulationController>;

class _AccessDataTOCache;
using AccessDataTOCache = boost::shared_ptr<_AccessDataTOCache>;
