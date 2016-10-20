#ifndef STARTSCREENCONTROLLER_H
#define STARTSCREENCONTROLLER_H

#include <QGraphicsView>

class QTimer;
class StartScreenController : public QObject
{
    Q_OBJECT
public:
    StartScreenController(QObject *parent);
    ~StartScreenController ();

    void runStartScreen (QGraphicsView* view);

private:
    void setupStartScreen (QGraphicsView* view);
    void setupTimer ();

private slots:
    void timeout ();

private:
    QGraphicsView* _view = 0;
    QGraphicsScene* _scene = 0;
    QGraphicsScene* _startScene;
    QTimer* _timer = 0;

};

#endif // STARTSCREENCONTROLLER_H
