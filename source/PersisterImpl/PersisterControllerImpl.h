#pragma once

#include <thread>

#include "PersisterWorker.h"
#include "PersisterInterface/PersisterController.h"

class _PersisterControllerImpl : public _PersisterController
{
public:
    void init(SimulationController const& simController);
    void shutdown();

    void saveSimulationToDisc(std::string const& filename, float const& zoom, RealVector2D const& center) override;

private:
    PersisterWorker _worker;
    std::thread* _thread = nullptr;
};
