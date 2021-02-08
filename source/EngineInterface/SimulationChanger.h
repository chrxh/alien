#pragma once

#include <QObject>

#include "Definitions.h"

class ENGINEINTERFACE_EXPORT SimulationChanger : public QObject
{
    Q_OBJECT
public:
    virtual ~SimulationChanger() = default;

    virtual void activate(SimulationParameters const& currentParameters) = 0;
    virtual void deactivate() = 0;
    Q_SLOT virtual void notifyNextTimestep() = 0;

    Q_SIGNAL void simulationParametersChanged();
    virtual SimulationParameters const& retrieveSimulationParameters() = 0;

};
