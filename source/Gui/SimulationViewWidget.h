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

    void updateScrollbars(IntVector2D const& worldSize, QVector2D const& center, double zoom);

    QGraphicsView* getGraphicsView() const;

    Q_SIGNAL void scrolledX(float centerX);
    Q_SIGNAL void scrolledY(float centerY);

private:
    Q_SLOT void horizontalScrolled();
    Q_SLOT void verticalScrolled();

    Ui::SimulationViewWidget *ui;

    boost::optional<double> _zoom;
};





