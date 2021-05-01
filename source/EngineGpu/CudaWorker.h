#pragma once

#include <windows.h>
#include <GL/gl.h>
#include <mutex>
#include <QThread>

#include "EngineInterface/ChangeDescriptions.h"
#include "EngineGpuKernels/AccessTOs.cuh"
#include "DefinitionsImpl.h"

class QOpenGLContext;
class QOffscreenSurface;

class CudaWorker : public QObject
{
    Q_OBJECT
public:
    CudaWorker(QObject* parent = nullptr);

    virtual ~CudaWorker();

    void init(
        SpaceProperties* space,
        int timestep,
        SimulationParameters const& parameters,
        CudaConstants const& cudaConstants,
        NumberGenerator* numberGenerator);
    void terminateWorker();
    bool isSimulationRunning();
    int getTimestep();
    void setTimestep(int timestep);
    void* registerImageResource(GLuint image);

    void addJob(CudaJob const& job);
    vector<CudaJob> getFinishedJobs(string const& originId);
    Q_SIGNAL void jobsFinished();

    Q_SIGNAL void timestepCalculated();

    Q_SIGNAL void errorThrown(QString message);

    Q_SLOT void run();

private:
    void processJobs();
    bool isTerminate();

private:
    CudaSimulation* _cudaSimulation = nullptr;
    NumberGenerator* _numberGenerator = nullptr;

    mutable std::mutex _mutex;
    std::condition_variable _condition;
    list<CudaJob> _jobs;
    vector<CudaJob> _finishedJobs;

    bool _simulationRunning = false;
    bool _terminate = false;
    boost::optional<int> _tpsRestriction;
    QOpenGLContext* _context;
    QOffscreenSurface* _surface;
};