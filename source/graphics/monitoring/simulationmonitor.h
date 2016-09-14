#ifndef SIMULATIONMONITOR_H
#define SIMULATIONMONITOR_H

#include <QMainWindow>

namespace Ui {
class SimulationMonitor;
}

class SimulationMonitor : public QMainWindow
{
    Q_OBJECT

public:
    explicit SimulationMonitor(QWidget *parent = 0);
    ~SimulationMonitor();

    void update (QMap< QString, qreal > data);

signals:
    void closed ();

protected:
    bool event(QEvent* event);

private:
    Ui::SimulationMonitor *ui;
};

#endif // SIMULATIONMONITOR_H
