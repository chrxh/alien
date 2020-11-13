#pragma once

#include <QObject>

#include "ModelBasic/Definitions.h"
#include "Web/Definitions.h"

class WebSimulationController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationController(WebAccess* webController, QWidget* parent = nullptr);

    void init(SimulationAccess* access);

    bool onConnectToSimulation();
    bool onDisconnectToSimulation(string const& simulationId, string const& token);

    optional<string> getCurrentSimulationId() const;
    optional<string> getCurrentToken() const;

private:
    Q_SLOT void checkIfSimulationImageIsRequired() const;

    optional<string> _currentSimulationId;
    optional<string> _currentToken;

    SimulationAccess* _access = nullptr;
    QWidget* _parent = nullptr;
    WebAccess* _webController = nullptr;
    QTimer* _timer = nullptr;
};