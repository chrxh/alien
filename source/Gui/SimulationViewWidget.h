#pragma once

#include <QWidget>
#include <QVector2D>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

namespace Ui {
	class SimulationViewWidget;
}

class SimulationViewWidget : public QWidget
{
    Q_OBJECT
public:
    SimulationViewWidget(QWidget* parent = nullptr);
    virtual ~SimulationViewWidget();

    QGraphicsView* getGraphicsView() const;

private:
    Ui::SimulationViewWidget *ui;
};





