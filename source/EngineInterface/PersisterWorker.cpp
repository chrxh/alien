#include "PersisterWorker.h"

void PersisterWorker::runThreadLoop()
{
    try {
        std::mutex mutexForLoop;
        std::unique_lock<std::mutex> lockForLoop(mutexForLoop);

        while (!_isShutdown.load()) {

        }
    } catch (std::exception const& e) {
        //#TODO
    }
}

void PersisterWorker::shutdown()
{
    _isShutdown = true;
}
