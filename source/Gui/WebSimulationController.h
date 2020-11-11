#pragma once

#include <QObject>

#include "Web/Definitions.h"

class WebSimulationController
    : public QObject
{
    Q_OBJECT
public:
    WebSimulationController(WebController* webController, QWidget* parent = nullptr);

    bool onConnectToSimulation();
    bool onDisconnectToSimulation(string const& simulationId, string const& token);

    optional<string> getCurrentSimulationId() const;
    optional<string> getCurrentToken() const;

private:
    optional<string> _currentSimulationId;
    optional<string> _currentToken;

    QWidget* _parent = nullptr;
    WebController* _webController = nullptr;
};