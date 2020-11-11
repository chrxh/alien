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
    bool onDisconnectToSimulation(string const& token);

    optional<string> getConnectionToken() const;

private:
    optional<string> _token;

    QWidget* _parent = nullptr;
    WebController* _webController = nullptr;
};