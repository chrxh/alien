#pragma once

#include <thread>

#include "EngineInterface/Definitions.h"
#include "EngineWorker.h"

#include "Definitions.h"

class _SimulationController
{
public:

    ENGINEIMPL_EXPORT void initCuda();

    ENGINEIMPL_EXPORT void newSimulation(
        IntVector2D size,
        int timestep,
        SimulationParameters const& parameters,
        GpuConstants const& gpuConstants);

    ENGINEIMPL_EXPORT void registerImageResource(GLuint image);

    ENGINEIMPL_EXPORT void getVectorImage(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);

    ENGINEIMPL_EXPORT void updateData(DataChangeDescription const& dataToUpdate);

    ENGINEIMPL_EXPORT void calcNextTimestep();
    ENGINEIMPL_EXPORT void runSimulation();
    ENGINEIMPL_EXPORT void pauseSimulation();

    ENGINEIMPL_EXPORT void closeSimulation();

    ENGINEIMPL_EXPORT IntVector2D getWorldSize() const;

    ENGINEIMPL_EXPORT boost::optional<int> getTpsRestriction() const;
    ENGINEIMPL_EXPORT void setTpsRestriction(boost::optional<int> const& value);

    ENGINEIMPL_EXPORT int getTps() const;

private:
    IntVector2D _worldSize;

    EngineWorker _worker;
    std::thread* _thread = nullptr;
};
