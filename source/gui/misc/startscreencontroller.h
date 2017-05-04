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

Q_SIGNALS:
    void startScreenFinished ();

private:
    void setupStartScene (QGraphicsView* view);
    void setupTimer ();

    void saveSceneAndView (QGraphicsView* view);
    void createSceneWithLogo ();
    void turnOffScrollbar ();

private Q_SLOTS:
    void timeout ();

private:
    bool isLogoTransparent () const;
    void scaleAndDecreaseOpacityOfLogo ();
    void restoreScene ();
    void turnOnScrollbarAsNeeded ();


private:
    QGraphicsView* _view = 0;
    QGraphicsScene* _savedScene = 0;
    QGraphicsScene* _startScene = 0;
    QGraphicsPixmapItem* _logoItem = 0;
    QMatrix _savedViewMatrix;
    QTimer* _timer = 0;

};

#endif // STARTSCREENCONTROLLER_H
