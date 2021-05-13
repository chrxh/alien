#include "ProgressBar.h"

#include <QWidget>
#include <QGraphicsView>
#include <QGraphicsRectItem>
#include <QTimer>

#include "Settings.h"
#include "QApplicationHelper.h"

namespace
{
    auto const ProgressBarWidth = 300;
}

ProgressBar::ProgressBar(std::string const& text, QWidget* parent)
    : QObject(parent)
{
    _graphicsView = new QGraphicsView(parent);
    auto rect = parent->geometry();
    auto posX = rect.width() / 2 - ProgressBarWidth / 2;
    auto posY = rect.height() - 220;
    _graphicsView->setFrameStyle(0);
    _graphicsView->setGeometry(posX, posY, ProgressBarWidth, 40);
    _graphicsView->show();
    _graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    _graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    auto scene = new QGraphicsScene(this);
    scene->setBackgroundBrush(Const::ProgressBarBackgroundColor);
    scene->setSceneRect(0, 0, 100, 100);

    _rectItem = new QGraphicsRectItem(QRectF(0.0f, 0.0f, 50.0f, 100.0f));
    _rectItem->setPen(Const::ProgressBarForegroundColor);
    _rectItem->setBrush(Const::ProgressBarForegroundColor);
    _rectItem->setPos(-50, 0);
    scene->addItem(_rectItem);

    auto progressTextItem = new QGraphicsSimpleTextItem(QString::fromStdString(text));
    auto font = progressTextItem->font();
    font.setPixelSize(15);
    progressTextItem->setFont(font);

    QFontMetrics metric(font);
    progressTextItem->setPos(50-metric.boundingRect(QString::fromStdString(text)).width() / 2, 7);
    progressTextItem->setBrush(Const::ProgressBarTextColor);
    scene->addItem(progressTextItem);

   _graphicsView->setScene(scene);

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &ProgressBar::progressStep);
    _timer->start(std::chrono::milliseconds(20));

    QApplicationHelper::processEventsForMilliSec(1000);
}

ProgressBar::~ProgressBar()
{
    delete _graphicsView;
}

void ProgressBar::progressStep()
{
    auto pos = _rectItem->pos();
    if (pos.x() > 150) {
        pos.setX(-100);
    } else {
        pos.setX(pos.x() + 2.0);
    }
    _rectItem->setPos(pos);
}
